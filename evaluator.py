from bert_score import score

def evaluate_answers(queries, answers):
    refs = ["Refer to the relevant arXiv paper(s) for confirmation."] * len(answers)
    P, R, F1 = score(answers, refs, lang="en", rescale_with_baseline=True)
    for i, (q, a, f1) in enumerate(zip(queries, answers, F1)):
        print(f"Q{i+1}: {q}\nAnswer: {a}\nFaithfulness Score (F1): {f1:.4f}\n")
