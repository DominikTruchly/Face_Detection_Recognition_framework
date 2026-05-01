\chapter{An appendix}
The created source code will be publicly available in the IS under the MIT license. The repository contains the following main files and directories:
\begin{figure}[htbp]
\makebox[0pt][c]{%
    \begin{minipage}{0.5\textwidth}
    \dirtree{%
    .1 code/.
    .2 src/.
    .3 evaluation.py.
    .3 metrics.py.
    .3 detection.py.
    .3 recognizer\_deepface.py.
    .3 recognizer\_insightface.py.
    .3 recognizer\_swin.py.
    .3 recognizer\_vit.py.
    .2 scripts/.
    .3 prepare\_cfp.py.
    .3 generate\_agedb\_protocol.py.
    .3 finetune\_classifier\_head.py.
    .3 recalculate\_metrics.py.
    .2 run\_detection.py.
    .2 run\_verification.py.
    .2 run\_identification.py.
    }
    \end{minipage}%
}
\caption{Directory structure of the published code repository.}
\end{figure}