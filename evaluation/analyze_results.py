import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="eval_results/llm_judgements_open_basic_lingshu_deepseek-v3.2-ca_final_data.json")
    args = parser.parse_args()

    with open(args.file_path, 'r', encoding='utf-8') as file:
        results = json.load(file)

    judgements = [data["judgement"] for data in results]

    conclusion_scores = list()
    conclusion_scores_sta = {
        "0": 0,
        "0.25": 0,
        "0.5": 0,
        "0.75": 0,
        "1.0": 0
    }

    process_scores = list()
    process_scores_sta = {
        "0~0.25": 0,
        "0.25~0.5": 0,
        "0.5~0.75": 0,
        "0.75~1": 0
    }

    process_failures = {
        "visual": 0,
        "interpretation": 0,
        "conclusion": 0
    }

    num_exp = 0
    error = dict()
    print(args.file_path)
    for i, jud in enumerate(judgements):
        score_con = float(jud["conclusion_score"]) / 4

        def find_index(num, ranges):
            return next((k for k in ranges if float(k) == num), None)

        index = find_index(score_con, conclusion_scores_sta)
        if index:
            conclusion_scores_sta[index] += 1
        else:
            error[f"Question[{i}]"] = f"Error conclusion score: {score_con}\n"

        exp_scores = []
        for exp in jud["experiments"]:
            num_exp += 1
            exp_step = 3

            vis = exp["visual_phenomenon"]
            inte = exp["interpretation"]
            con = exp["sub-conclusion"]

            if vis == -1:
                vis = 1
                exp_step = 2

            if vis == 0:
                process_failures["visual"] += 1

            if vis == 1 and inte == 0:
                process_failures["interpretation"] += 1

            if vis == 1 and inte == 1 and con == 0:
                process_failures["conclusion"] += 1

            if exp_step == 3:
                exp_scores.append((vis + vis * inte + vis * inte * con) / exp_step)
            elif exp_step == 2:
                exp_scores.append((inte + vis * inte * con) / exp_step)
            else:
                raise ValueError

        score_pro = sum(exp_scores) / len(exp_scores)

        def find_range(num, ranges):
            return next((k for k in ranges if float(k.split('~')[0]) <= num < float(k.split('~')[1])), None)

        range = find_range(score_pro, process_scores_sta)
        if range:
            process_scores_sta[range] += 1
        elif score_pro == 1.0:
            process_scores_sta["0.75~1"] += 1
        else:
            if f"Question[{i}]" in error:
                error[f"Question[{i}]"] += f"Error process score: {score_pro}\n"
            else:
                error[f"Question[{i}]"] = f"Error process score: {score_pro}\n"

        conclusion_scores.append(score_con)
        process_scores.append(score_pro)
        # add score to results
        results[i]["conc_score"] = score_con
        results[i]["proc_score"] = score_pro

    lcr = []
    for c_score, p_score in zip(conclusion_scores, process_scores):
        if c_score + p_score == 0:
            lcr.append(0)
        else:
            lcr.append(2 * c_score * p_score / (c_score + p_score))
    mean_pro = sum(process_scores) / len(process_scores)
    mean_con = sum(conclusion_scores) / len(conclusion_scores)
    mean_lcr = sum(lcr) / len(lcr)

    static_results = {
        "conclusion_scores_sta": conclusion_scores_sta,
        "process_scores_sta": process_scores_sta,
        "process_failures": process_failures,
        "num_exp": num_exp,
        "error": error,
        "conclusion_scores": conclusion_scores,
        "process_scores": process_scores,
    }
    static_results['lcrs_scores'] = mean_lcr
    static_results['proc_scores'] = mean_pro
    static_results['conc_scores'] = mean_con

    save_path = f"./sta_result/STA_{args.file_path.split('/')[-1]}"
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(static_results, file, ensure_ascii=False, indent=2)
    # save the updated results with scores
    updated_results_path = f"./sta_result/Updated_{args.file_path.split('/')[-1]}"
    with open(updated_results_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)