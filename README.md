# MuNEKG_SO
Stack Overflow users often share associated informative links (i.e., internal or external websites) in their posts. Prior studies
have applied these internal links to construct a knowledge graph with simply combing the question and its associated answers as
a entity to mining the Software Engineering (i.e., SE) knowledge. However, the simple knowledge graph could not comprehensively
represent the complex interactive knowledge and abundance of relevant external resources. In this study, we first quantitatively and
qualitatively investigate the actual characteristics of the complex knowledge that are included in the knowledge graph for a specific
question thread. We find that a specific question thread includes complex knowledge structures (i.e., hierarchy, coupling and complete)
which are informative and different intended roles for the crowd knowledge of a question thread. Our findings highlight that it is
necessary to propose a fine-grained knowledge graph to comprehensively represent the complex knowledge network. Therefore, based
on our findings, we propose a multi-nodes multi-edges knowledge graph, whose entities include questions, answers, and external
resources, and the edges include the hierarchy (i.e., Q-A), coupling(i.e., duplicate, concatenation, containment, pre-knowledge, and
post-knowledge) and complete (i.e., working example, supporting information, reference existing similar questions relations) structures.
Meanwhile, we develop a automated generation of the multi-nodes multi-edges knowledge graph approach MuNEKG for a specific
question thread, which contains two phases of identifying the entities and identifying the edges. Through the extensive case study,
MuNEKG can achieve above 85% accuracy of all the types of edges and above 93% Top-2 accuracy. Furthermore, to validate the
effectiveness of our proposed multi-nodes multi-edges knowledge graph, we develop a automated generated best answers application
for developers to quickly find their wanted answers. Our user studies with 100 Java questions confirm the effectiveness of automated
generated best answers application. Finally, we also discuss the implications of our findings for developers, researchers and Stack
Overflow moderators.
