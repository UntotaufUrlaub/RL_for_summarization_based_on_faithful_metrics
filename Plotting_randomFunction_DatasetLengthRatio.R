library(ggplot2)
library(data.table)
library(quanteda)
randoms <- fread("~/Desktop/Storage/uni/22SoSe/MA/metric_result_saves/aggrefact_auswertungen_saves/aggrefact_randoms.csv")
randoms.small <- randoms[, c("cnndm", "xsum")]
randoms.mt <- melt(randoms.small)


ggplot(randoms.mt, aes(x = value)) + geom_histogram() + facet_grid(variable ~ .)


aggrefact <- fread("~/Desktop/Storage/uni/22SoSe/MA/Code/human_alignment/AggreFact-main/data/aggre_fact_final.csv")
aggrefact.small <- aggrefact[, c("origin", "doc", "summary")]
aggrefact.small[,DocLen := nchar(doc)]
aggrefact.small <- aggrefact.small[DocLen < 20000]
aggrefact.small[,SummLen := nchar(summary)]
aggrefact.small[,SummLenInSent := nsentence(summary)]
aggrefact.small[,DocLenInSent := nsentence(doc)]
aggrefact.small[,Ratio := SummLen / DocLen]
aggrefact.small[,RatioInSent := SummLenInSent / DocLenInSent]
ggplot(aggrefact.small, aes(x = DocLen)) + geom_histogram() + facet_grid(origin ~ .)
ggplot(aggrefact.small, aes(x = SummLen)) + geom_histogram() + facet_grid(origin ~ .)
ggplot(aggrefact.small, aes(x = SummLenInSent)) + geom_histogram() + facet_grid(origin ~ .)
ggplot(aggrefact.small, aes(x = Ratio)) + geom_histogram() + facet_grid(origin ~ .) + xlim(c(0, 0.5))
ggplot(aggrefact.small, aes(x = origin, y = Ratio)) + geom_violin()+ ylim(c(0, 0.3))
ggplot(aggrefact.small, aes(x = origin, y = Ratio)) + geom_boxplot() + ylim(c(0, 0.3))
ggplot(aggrefact.small, aes(x = RatioInSent)) + geom_histogram() + facet_grid(origin ~ .) + xlim(c(0, 0.75))
