import random
import re

import torch
from torch import nn
import copy
import matplotlib.pyplot as plt
from get_metrics_lookup import get_metrics_lookup
from human_allignment import eval_metric
import sys
import numpy as np
import pickle

import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self, family, use_input_normalization):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(48, 20),  # todo: input länge anpassen.
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.birthday = -1
        self.family = family
        self.use_input_normalization = use_input_normalization

    def forward(self, x):
        if self.use_input_normalization:
            x = nn.functional.normalize(x, dim=0)
        return self.linear_relu_stack(x)


def set_to_one_feature(model, feature_number):
    print(f"create model for feature {feature_number}")
    with torch.no_grad():
        for j in range(0, len(model.linear_relu_stack), 2):
            if j == 0:
                value = 0
            else:
                value = 1

            if model.use_input_normalization:
                # print(f"set bias {value} for layer {j}")
                model.linear_relu_stack[j].bias.data.fill_(value)
            else:
                # print(f"set bias 0 for layer {j}")
                model.linear_relu_stack[j].bias.data.fill_(0)

            # print(f"set all weights to {value} for layer {j}")
            for i in range(len(model.linear_relu_stack[j].weight)):
                for x in range(len(model.linear_relu_stack[j].weight[i])):
                    model.linear_relu_stack[j].weight[i][x] = value

        for i in range(len(model.linear_relu_stack[0].weight)):
            model.linear_relu_stack[0].weight[i][feature_number] = 1


def mutate(model_in, birthday, escalation_power, mutation_power=0.01):
    m = copy.deepcopy(model_in)
    m.birthday = birthday
    # print(" --- old --- ")
    # for name, param in m.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    age = m.birthday - model_in.birthday + 1
    if random.random() > 0.5:
        m_power = mutation_power + escalation_power * age ** 2
    else:
        m_power = mutation_power / (1 + (escalation_power / mutation_power) * age)
    # print("m power factor:" + str(m_power / mutation_power))

    with torch.no_grad():
        for param in m.parameters():
            param.data += m_power * torch.randn_like(param)

    # print(" --- new --- ")
    # for name, param in m.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    return m


def model_to_string(model_in):
    res = ""
    for param in model_in.parameters():
        res += str(param.data) + " "
    res += str(model_in.use_input_normalization)
    return res


class Training:
    def __init__(self):
        self.saving_path = "Training_save.pkl"
        self.fitness_cache = {}
        self.validation_cache = {}
        self.data_portion_gpt = 1
        self.data_portion_rlfhf = 0.1
        self.data_portion_rlfhf_axes = 0.5
        self.data_portion_val_gpt = 1
        self.data_portion_val_rlfhf = 0.5
        self.data_portion_val_rlfhf_axes = 1
        self.metrics_lookup_train = get_metrics_lookup(self.data_portion_gpt, self.data_portion_rlfhf,
                                                       self.data_portion_rlfhf_axes, 'train')
        self.metrics_lookup_val = get_metrics_lookup(self.data_portion_val_gpt, self.data_portion_val_rlfhf,
                                                     self.data_portion_val_rlfhf_axes, 'val')
        self.metric_names = [
            'my_dae_batched',
            'my_questeval024_batched',
            'my_questeval011_fscore',
            'my_questeval011_precision',
            'my_questeval011_recall',
            'my_summacZS_batched',
            'my_summacConv_batched',
            'rouge1',
            'rouge2',
            'rougeL',
            'rougeLsum',
            'rouge1_stem',
            'rouge2_stem',
            'rougeL_stem',
            'rougeLsum_stem',
            'length',
            # 'negative_length',    # No: negation is no new info.
            'length_share',
            'compression',
            'grusky_Coverage',
            'grusky_Density',
            'grusky_Compression',
            'blanc_help',
            # 'blanc_tune',   # No
            'disco_score_SENT_NN_truncate',
            'disco_score_SENT_Entity_truncate',
            'disco_score_FOCUS_NN_truncate',
            'disco_score_FOCUS_ENTITY_truncate',
            'disco_score_SENT_NN_longformer',
            'disco_score_SENT_Entity_longformer',
            'disco_score_FOCUS_NN_longformer',
            'disco_score_FOCUS_ENTITY_longformer',
            'BertScore_Precision',
            'BertScore_Recall',
            'BertScore_F1',
            'ctc_D-cnndm_consistency',
            'ctc_D-cnndm_relevance',
            'ctc_D-xsum_consistency',
            'SGfLM_ShannonScore',
            'SGfLM_InfoDiff',
            'SGfLM_BlancShannon',
            'psudo_perplexity_bert_lefthandContext_normal',
            # 'psudo_perplexity_bert_lefthandContext_negated',    # No: negation is no new info.
            'psudo_perplexity_bert_fullContext_normal',
            # 'psudo_perplexity_bert_fullContext_negated',    # No: negation is no new info.
            'psudo_perplexity_bert_lefthandContext_summaryOnly_normal',
            # 'psudo_perplexity_bert_lefthandContext_summaryOnly_negated',    # No: negation is no new info.
            'psudo_perplexity_bert_fullContext_summaryOnly_normal',
            # 'psudo_perplexity_bert_fullContext_summaryOnly_negated',    # No: negation is no new info.
            'psudo_perplexity_bert_lefthandContext_swapped_normal',
            # 'psudo_perplexity_bert_lefthandContext_swapped_negated',    # No: negation is no new info.
            # 'psudo_perplexity_bert_fullContext_swapped_normal', # No
            # 'psudo_perplexity_bert_fullContext_swapped_negated', # No
            'estim_soft',
            'estim_coherence',
            'estim_allTokens_alarms',
            'estim_notAllTokens_alarms',
            # 'chatgpt-zs_defaultFalse', # No
            # 'chatgpt-zs_defaultTrue'  # No
        ]

        self.population_count = 104
        self.include_sigle_feature_models = True
        self.input_normalisation = [True, False]
        self.share_to_keep = 2
        self.duration = 101
        self.loop_marker = 0
        self.plot_interval = 1
        self.plot_pause = 0.5
        self.mutation_power = 0.04
        self.escalation = 0.004

        metrics = {}
        for name in self.metric_names:
            def create_function(metric_name):
                return lambda summ, doc: self.metrics_lookup_train[doc][summ][metric_name + "_score"]

            metrics[name] = create_function(name)
        self.baselines = eval_metric(metrics, ['train', ], 'example', self.data_portion_gpt, self.data_portion_rlfhf,
                                     self.data_portion_rlfhf_axes)

        metrics = {}
        for name in self.metric_names:
            def create_function(metric_name):
                return lambda summ, doc: self.metrics_lookup_val[doc][summ][metric_name + "_score"]

            metrics[name] = create_function(name)
        self.val_baselines = eval_metric(metrics, ['val', ], 'example', self.data_portion_val_gpt,
                                         self.data_portion_val_rlfhf,
                                         self.data_portion_val_rlfhf_axes)
        # self.val_baselines = None

        print("Training baselines")
        self._best_baseline = print_eval_res(self.baselines)[0]
        print("----")
        print("Val baselines")
        self._best_val_baseline = print_eval_res(self.val_baselines)[0]
        print("----")

        self.family_count = 0
        self.family_change_markers = []

        self.pop = [None] * self.population_count
        for i in range(self.population_count):
            norm = random.choice(self.input_normalisation)
            self.pop[i] = NeuralNetwork(str(self.family_count) + "-" + str(norm) + "-" + str(i), norm).to(device)

        if self.include_sigle_feature_models:
            for norm in self.input_normalisation:
                feature_amount = 48
                self.pop.extend([None] * feature_amount)
                for i in range(feature_amount):
                    n = NeuralNetwork(str(self.family_count) + "--" + self.metric_names[i] + "-" + str(norm) + "--0", norm)
                    set_to_one_feature(n, i)
                    self.pop[self.population_count + i] = n.to(device)
                self.population_count = self.population_count + feature_amount

        self.learning_curve = [None] * self.duration
        self.validation_curve_y = []
        self.validation_curve_x = []

        self.best_val_score = None
        self.best_val_model = None

        self._text_plot_train_curve_precision = 3
        self._text_plot_curve_length = 20

        assert self.population_count % self.share_to_keep == 0, "population count has to be devidably by share_to_keep without remainder"

    def _text_plot(self, step, families, birthdays, norms):
        print(f"---- Plot: Step {step}")

        lc = [int(elem*10**self._text_plot_train_curve_precision) for elem in self.learning_curve if elem is not None]
        lc_skip = len(lc) // self._text_plot_curve_length + 1
        # print("lc skip: ", lc_skip)
        lc = [lc[i] for i in range(len(lc)) if i % lc_skip == 0 or i == len(lc)-1]
        print("learning curve: \t\t" + str(lc))
        print(f"T baseline:\t\t\t{int(self._best_baseline[0]*10**self._text_plot_train_curve_precision)}\t\t{self._best_baseline[1]}")

        vc = [int(elem * 10 ** self._text_plot_train_curve_precision) for elem in self.validation_curve_y]
        vc_skip = len(vc) // self._text_plot_curve_length + 1
        # print("vc skip: ", vc_skip)
        vc = [vc[i] for i in range(len(vc)) if i % vc_skip == 0 or i == len(vc) - 1]
        print("val curve: \t\t\t" + str(vc))
        print(f"V baseline:\t\t\t{int(self._best_val_baseline[0] * 10 ** self._text_plot_train_curve_precision)}\t\t{self._best_val_baseline[1]}")

        print("family changes: \t\t" + str(self.family_change_markers))

        if families is None:
            print("family set: \t\t\t-")
            print("families: \t\t\t-")
        else:
            seen = set()
            family_set = [x for x in families if not (x in seen or seen.add(x))]    # keeps the order
            family_set_length = len(family_set)
            if family_set_length != len(families):
                print("family set: \t\t\t(" + str(family_set_length) + ")\t" + str(family_set))
            else:
                print("family set: \t\t\t(==)\t")
            print("families: \t\t\t" + str(families))

        if birthdays is None:
            print("birth range: \t\t\t-")
            print("births: \t\t\t-")
        else:
            print("birth range: \t\t\t(" + str(min(birthdays)) + ", " + str(max(birthdays)) + ")")
            print("births: \t\t\t" + str(birthdays))

        if norms is not None:
            trues = sum(norms)
            falses = len(norms) - trues
            if falses == 0 or trues == 0:
                norm_string = ""
            else:
                norm_string = str(norms)
            print("Using input norma.: \t\t(" + str(trues) + ", " + str(falses) + ")\t" + norm_string)
        else:
            print("Using input norma.: \t\t-")

    def _train(self, population_count, share_to_keep, duration, plot_interval, plot_pause, fitness_function, baselines,
               val_baselines, validation_interval, validation_function, saving_interval, text_plot):
        print("start _train")

        if plot_interval != 0:
            if not text_plot:
                # plt.ion()
                fig = plt.figure()
                plt.xlim([0, len(self.learning_curve)])
                plt.ylim([-0.1, 1.1])
                plt.title('Correlation Learning Plot')
                line, = plt.plot(self.learning_curve)
                validation_line, = plt.plot(self.validation_curve_x, self.validation_curve_y)
                age_text = plt.text(0, -0.22, "ages:")
                family_text = plt.text(len(self.learning_curve)/2, -0.22, "families:")
                plt.scatter(self.family_change_markers, [0]*len(self.family_change_markers), color="g")
                for _, v in baselines.items():
                    plt.axhline(y=abs(v), color='paleturquoise', linestyle='--', zorder=0)
                for _, v in val_baselines.items():
                    plt.axhline(y=abs(v), color='moccasin', linestyle='--', zorder=0)
                plt.pause(plot_pause)
            else:
                self._text_plot(str(self.loop_marker) + " Initial", None, None, None)

        for i in range(self.loop_marker, duration):
            self.loop_marker = i + 1
            scores = [0]*len(self.pop)
            for j in range(len(self.pop)):
                cache_key = model_to_string(self.pop[j])
                if cache_key not in self.fitness_cache:
                    self.fitness_cache[cache_key] = fitness_function(self.pop[j])
                # else:
                #     print("use fitness cache")
                scores[j] = self.fitness_cache[cache_key]

            #print("scores:", scores)
            #for a, b in zip(scores, self.pop):
            #    print("zip:", a, b)
            best = [x for _, x in sorted(zip(scores, self.pop), reverse=True, key=lambda t: abs(t[0]))][:self.population_count // self.share_to_keep]
            self.learning_curve[i] = sorted(scores, reverse=True, key=lambda t: abs(t))[0]

            family_set = set([m.family for m in best])
            if len(family_set) == 1:
                self.family_count += 1
                self.family_change_markers.append(i)
                if plot_interval != 0 and not text_plot:
                    plt.scatter(i, 0, color="g")

                for k in range(len(best)):
                    match = re.match(r"[0-9]+--(.*)--[0-9]+", best[k].family)
                    if match:
                        best[k].family = str(self.family_count) + "--" + match.group(1) + "--" + str(k)
                    else:
                        best[k].family = str(self.family_count) + "-" + str(best[k].use_input_normalization) + "-" + str(k)

            if i % validation_interval == 0:
                cache_key = model_to_string(best[0])
                if cache_key not in self.validation_cache:
                    self.validation_cache[cache_key] = validation_function(best[0])

                self.validation_curve_y.append(self.validation_cache[cache_key])
                self.validation_curve_x.append(i)
                if plot_interval != 0 and not text_plot:
                    validation_line.set_ydata(self.validation_curve_y)
                    validation_line.set_xdata(self.validation_curve_x)

                if self.best_val_score is None or self.best_val_score < self.validation_cache[cache_key]:
                    self.best_val_score = self.validation_cache[cache_key]
                    self.best_val_model = best[0]

            if plot_interval != 0 and i % plot_interval == 0:
                birthdays = [m.birthday for m in best]
                families = [m.family for m in best]
                norms = [m.use_input_normalization for m in best]
                if not text_plot:
                    age_text.set_text("ages: " + str(birthdays))
                    family_text.set_text("families: " + str(families))

                    line.set_ydata(self.learning_curve)
                    fig.canvas.draw()
                    plt.pause(plot_pause)
                else:
                    self._text_plot(i, families, birthdays, norms)

            for j in range(len(best)):
                for k in range(share_to_keep):
                    if k == 0:
                        new_model = best[j]
                    else:
                        new_model = mutate(best[j], i, self.escalation)
                    self.pop[j*share_to_keep+k] = new_model

            if i + 1 % saving_interval:
                with open(self.saving_path, 'wb') as file:
                    pickle.dump(self, file)

        if plot_interval != 0 and not text_plot:
            # plt.ioff()
            plt.close()

    def evaluate(self, model_in, metrics_lookup, split, dp_gpt, dp_rlfhf, dp_rlfhf_axes, features_list_cache):
        def function(summs, docs):
            key = str(summs)+str(docs)
            if key not in features_list_cache:
                print("prepare features")
                features_list = []
                for j in range(len(summs)):
                    doc = docs[j]
                    summ = summs[j]
                    row = metrics_lookup[doc][summ]
                    features = [0] * len(self.metric_names)
                    for i in range(len(self.metric_names)):
                        features[i] = row[self.metric_names[i] + "_score"]
                    features_list.append(features)
                features_list_cache[key] = torch.tensor(features_list).to(device)

            with torch.no_grad():
                return model_in(features_list_cache[key]).detach().cpu().numpy().ravel()

        return eval_metric(
            {"res": function},
            [split, ],
            'example',
            dp_gpt, dp_rlfhf, dp_rlfhf_axes)["res"]

    def train(self, saving_interval, text_plot):
        features_list_cache = {}
        self._train(self.population_count, self.share_to_keep, self.duration, self.plot_interval, self.plot_pause,
                    lambda m: self.evaluate(
                        m,
                        self.metrics_lookup_train,
                        "train",
                        self.data_portion_gpt,
                        self.data_portion_rlfhf,
                        self.data_portion_rlfhf_axes,
                        features_list_cache),
                    self.baselines, self.val_baselines, validation_interval=1, # todo: anpassen.

                    validation_function=lambda m: self.evaluate(
                        m,
                        self.metrics_lookup_val,
                        "val",
                        self.data_portion_val_gpt,
                        self.data_portion_val_rlfhf,
                        self.data_portion_val_rlfhf_axes,
                        features_list_cache),
                    saving_interval=saving_interval,
                    text_plot=text_plot)


def print_eval_res(dict, tap_size=8, width=17):
    lines = []
    for k, v in dict.items():
        lines.append((abs(v), k))
    lines.sort(reverse=True)
    for v, k in lines:
        print(k + (width - int(len(k) / tap_size)) * "\t", dict[k],)
    return lines


start = time.time()

trainer = Training()
trainer.train(4, text_plot=True)

time_duration = time.time() - start
print(f"Training took {time_duration} secs. {time_duration / trainer.duration} per update step")

# with open("Training_save.pkl", 'rb') as file:
#     trainer = pickle.load(file)
# trainer.duration = 1000
# trainer.learning_curve.extend([None] * (trainer.duration-len(trainer.learning_curve)))
# trainer.plot_interval = 999
# trainer.plot_pause = 120
# trainer.train(20)

# with open("Training_save_server2.pkl", 'rb') as file:
#     trainer = pickle.load(file)
# trainer.duration = 1
# trainer.plot_pause = 120
# trainer.train(10, text_plot=True)
