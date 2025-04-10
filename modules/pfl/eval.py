from tqdm import tqdm
import torch


def evaluate(data_loader, model, evaluator):

    return_scores_by_sample = True
    return_answers = True

    if return_scores_by_sample:
        scores_by_samples = {}
        total_accuracies = []
        total_anls = []

    else:
        total_accuracies = 0
        total_anls = 0

    all_pred_answers = []
    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch['question_id'])
        with torch.no_grad():
            outputs, pred_answers, answer_conf = model.forward(batch, return_pred_answer=True)

        metric = evaluator.get_metrics(batch['answers'], pred_answers, batch.get('answer_type', None))

        if return_scores_by_sample:
            for batch_idx in range(bs):
                scores_by_samples[batch['question_id'][batch_idx]] = {
                    'question': batch['questions'][batch_idx],
                    'answer': batch['answers'][batch_idx][0],
                    'accuracy': metric['accuracy'][batch_idx],
                    'anls': metric['anls'][batch_idx],
                    'predicted_answer': pred_answers[batch_idx],
                    'confidence': answer_conf[batch_idx],
                    'loss': outputs.loss.item()
                }

        if return_scores_by_sample:
            total_accuracies.extend(metric['accuracy'])
            total_anls.extend(metric['anls'])

        else:
            total_accuracies += sum(metric['accuracy'])
            total_anls += sum(metric['anls'])

        if return_answers:
            all_pred_answers.extend(pred_answers)

    if not return_scores_by_sample:
        total_accuracies = total_accuracies/len(data_loader.dataset)
        total_anls = total_anls/len(data_loader.dataset)
        scores_by_samples = []
    
    return total_accuracies, total_anls, all_pred_answers, scores_by_samples


