%%% ====================================================
%%% @LaTeX-file{
%%%    filename  = "TCAM.cls",
%%%    date      = "11/02/2026",
%%%    author    = "Fabricio Simeoni de Sousa and Ana Claudia Ribeiro",
%%%    abstract  = "Class for TCAM authors.",
%%% }
%%% ====================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TCAM}

% Flag: assume inicialmente que nenhuma opção foi passada
\newif\ifTCAM@nooptions
\TCAM@nooptionstrue

% Opções da classe
\DeclareOption{english}{%
  \TCAM@nooptionsfalse
  \PassOptionsToPackage{english}{babel}%
}
\DeclareOption{brazil}{%
  \TCAM@nooptionsfalse
  \PassOptionsToPackage{brazil}{babel}%
}

% Todas as opções não tratadas explicitamente
\DeclareOption*{%
  \TCAM@nooptionsfalse
  \PassOptionsToClass{\CurrentOption}{article}% por exemplo
}

% Processa opções
\ProcessOptions\relax

% Se nenhuma opção foi passada, escolhe um padrão
\ifTCAM@nooptions
  % aqui você sabe que NENHUMA opção foi usada na classe
  \PassOptionsToPackage{english}{babel}% idioma padrão
\fi

\RequirePackage{babel}

\RequirePackage{amsmath,amsthm}
\RequirePackage{ifthen}

\LoadClass[11pt, a4paper]{article}
\usepackage[top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}
\usepackage{setspace}
\onehalfspacing

\renewcommand{\encodingdefault}{T1}

\renewcommand{\theequation}{\thesection.\arabic{equation}}


\usepackage{titlesec}
\titleformat{\section}
            {\normalfont\Large\bfseries}
            {\thesection.\setcounter{equation}{0}}{2mm}{}
%%                                \setcounter{thmTCAM}{0}
%%                                \setcounter{teoTCAM}{0}
%%                                \setcounter{lemmaTCAM}{0}
%%                                \setcounter{lemaTCAM}{0} 
%%                                \setcounter{defTCAMi}{0}
%%                                \setcounter{defTCAMp}{0} 
%%                                \setcounter{propTCAMi}{0}
%%                                \setcounter{propTCAMp}{0}
%                                }{-20mm}{}
%
\titleformat{\subsection}
            {\normalfont\large\bfseries}{\thesubsection.}{2mm}{}
\titleformat{\subsubsection}
            {\normalfont\normalsize\bfseries}{\thesubsubsection.}{1mm}{}


% legacy
%\newcommand{\newsec}[1]{\section{#1}\setcounter{equation}{0} 
%                                    \setcounter{thmTCAM}{0}
%                                    \setcounter{teoTCAM}{0}
%                                    \setcounter{lemmaTCAM}{0}
%                                    \setcounter{lemaTCAM}{0} 
%                                    \setcounter{defTCAMi}{0}
%                                    \setcounter{defTCAMp}{0} 
%                                    \setcounter{propTCAMi}{0}
%                                    \setcounter{propTCAMp}{0} }            
                     

\theoremstyle{plain}
\newtheorem{theorem}{\theoremname}[section]
\newtheorem{lemma}[theorem]{\lemmaname}
\newtheorem{corollary}[theorem]{\corollaryname}
\newtheorem{proposition}[theorem]{\propositionname}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{\definitionname}


% ---------- Language‑dependent names ----------

% English
\addto\captionsenglish{%
  \def\theoremname{Theorem}%
  \def\lemmaname{Lemma}%
  \def\corollaryname{Corollary}%
  \def\propositionname{Proposition}%
  \def\definitionname{Definition}%
  \def\abstracttcamname{Abstract}%
  \def\keywordsname{Keywords}%
  \def\runningtitlesname{Running title}%
  \def\acknowledgementsname{Acknowledgements}%
}

% Brazilian Portuguese (babel option "brazil" or "brazilian")
\addto\captionsbrazil{%
  \def\theoremname{Teorema}%
  \def\lemmaname{Lema}%
  \def\corollaryname{Corolário}%
  \def\propositionname{Proposição}%
  \def\definitionname{Definição}%
  \def\abstracttcamname{Resumo}%
  \def\keywordsname{Palavras-chave}%
  \def\runningtitlesname{Título resumido}%
  \def\acknowledgementsname{Agradecimentos}%
}
% If you also allow "brazilian" as babel option:
\addto\captionsbrazilian{%
  \def\theoremname{Teorema}%
  \def\lemmaname{Lema}%
  \def\corollaryname{Corolário}%
  \def\propositionname{Proposição}%
  \def\definitionname{Definição}%
  \def\abstracttcamname{Resumo}%
  \def\keywordsname{Palavras-chave}%
  \def\runningtitlesname{Título resumido}%
  \def\acknowledgementsname{Agradecimentos}%
}

% Optionally set a default main language here, or leave it to the document.
%\AtBeginDocument{\selectlanguage{english}}


%
%
%
%\newtheorem{thmTCAM}{Theorem}
%\renewcommand{\thethmTCAM}{\arabic{section}.\arabic{thmTCAM}}
%\newtheorem{teoTCAM}{Teorema}
%\renewcommand{\theteoTCAM}{\arabic{section}.\arabic{teoTCAM}}
%
%\newtheorem{lemmaTCAM}{Lemma}
%\renewcommand{\thelemmaTCAM}{\arabic{section}.\arabic{lemmaTCAM}}
%\newtheorem{lemaTCAM}{Lema}
%\renewcommand{\thelemaTCAM}{\arabic{section}.\arabic{lemaTCAM}}
%
%\newtheorem{propTCAMi}{Proposition}
%\renewcommand{\thepropTCAMi}{\arabic{section}.\arabic{propTCAMi}}
%\newtheorem{propTCAMp}{Proposi\c{c}\~{a}o}
%\renewcommand{\thepropTCAMp}{\arabic{section}.\arabic{propTCAMp}}
%
%\newtheorem{defTCAMi}{Definition}
%\renewcommand{\thedefTCAMi}{\arabic{section}.\arabic{defTCAMi}}
%\newtheorem{defTCAMp}{Defini\c{c}\~{a}o}
%\newtheorem{remTCAMp}{Observa\c{c}\~{a}o}
%\renewcommand{\thedefTCAMp}{\arabic{section}.\arabic{defTCAMp}}
%
%\newtheorem{coroTCAMi}{Corollary}
%\renewcommand{\thecoroTCAMi}{\arabic{section}.\arabic{thmTCAM}}
%\newtheorem{coroTCAMp}{Corol\'ario}
%\renewcommand{\thecoroTCAMp}{\arabic{section}.\arabic{teoTCAM}.\arabic{coroTCAMp}}


%\renewenvironment{abstract}
%{\begin{list}{}{%
%  \setlength{\rightmargin}{0.5cm}
%  \setlength{\leftmargin}{0.5cm}
%  \small} 
%  \item[] }
%{\end{list}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\maketitle{\par
    \begingroup
      \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
      \long\def\@makefntext##1{\parindent 1em\noindent
              \hb@xt@0em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%      
          \@maketitle      
      \thispagestyle{plain}\@thanks
    \endgroup
    \setcounter{footnote}{0}%
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\title\relax
    \global\let\author\relax
    \global\let\date\relax
    \global\let\and\relax
  }

\def\runningtitle#1{\def\@runningtitle{#1}}
\def\keywords#1{\def\@keywords{#1}}
\def\abstracttcam#1{\def\@abstracttcam{#1}}
\def\acknowledgments#1{\def\@acknowledgments{#1}}
\def\aff#1{\def\@aff{#1}}

\renewcommand\aff{\footnote}

\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
  \end{center}%
    %\vskip 1em%
		\setlength{\rightmargin}{2cm}
		\setlength{\leftmargin}{2cm}
		%\raggedright
		{\small {\bf \abstracttcamname: }\@abstracttcam \par}%
		\vskip 1em%
		{\small {\bf \keywordsname: }\@keywords \par}%
		\vskip 1em%	
		{\small {\bf \runningtitlesname: }\@runningtitle \par}%
		\vskip 1em%
		{\small {\bf \acknowledgementsname: }\@acknowledgments \par}%
		\setlength{\rightmargin}{0cm}
		\setlength{\leftmargin}{0cm}
  \par
  \vskip 1.5em}
  
\def\abstracttcamenglish#1{\def\@abstracttcamenglish{#1}}%
\def\titleenglish#1{\def\@titleenglish{#1}}%
\def\keywordsenglish#1{\def\@keywordsenglish{#1}}%


\newcommand{\insertenglishinformation}{
	\vspace{5mm}
    \noindent\textbf{\@titleenglish}\\[3mm]
    \noindent\textbf{Abstract:} {\@abstracttcamenglish}\\[3mm]
    \noindent\textbf{Keywords:} {\@keywordsenglish}
}

\ifthenelse{\equal{\languagename}{brazil}}{%
    % Texto para português
    \par\bigskip
    \AtEndDocument{\insertenglishinformation}%
  }{%
  }%

\setcounter{page}{1}

\endinput
