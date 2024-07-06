# Characterizing Signaling Dependent Programs During Embryonic Cell-fate Decisions in vivo

## Background

During embryonic development, [cell fate](https://en.wikipedia.org/wiki/Cell_fate_determination) determination relies heavily on interactions between cells and their environment. Classical theories such as the [morphogen theory](https://en.wikipedia.org/wiki/Morphogen) propose that cell fate is determined by gradients of [signaling molecules](https://en.wikipedia.org/wiki/Cell_signaling). However, these theories face challenges in complex 3D models like the [early mouse embryo](https://www.emouseatlas.org/emap/ema/theiler_stages/StageDefinition/stagedefinition.html). Current research relies on [knockout experiments](https://en.wikipedia.org/wiki/Knockout_mouse) due to difficulties in studying low signaling molecule concentrations and cell movements. Recent [in vitro](https://en.wikipedia.org/wiki/In_vitro) experiments using single-cell RNA sequencing have shown that increasing BMP [Bone Morphogenetic Protein](https://en.wikipedia.org/wiki/Bone_morphogenetic_protein) concentration amplifies immediate cellular responses without altering gene targets significantly. BMP signals are critical during gastrulation for forming various cell types. Interestingly, identical signaling environments can lead to diverse cell fates, suggesting that immediate responses vary based on cell state influenced by various factors.

## Project Goals

The main objective of this project is to investigate how immediate cell state responses vary with BMP concentration. Additionally, I aim to develop predictive models for BMP-influenced cell fate decisions and explore differences in BMP member perception in vivo. This study will utilize multiple single-cell technologies to analyze embryos exposed to different BMP concentrations, leveraging the reproducible system of gastrulating embryos.

## Data Processing

In a preliminary experiment, I dissected and individually cultured ten E7.5 mouse embryos (one week post-fertilization) for 2 hours. Among these embryos, five were cultured in basic media (control), three in media supplemented with 1 µM BMP-4, and two in media with 2 µM BMP-4. Following the culture period, single cells were carefully sorted into a 384-well plate, and their RNA was sequenced. Initial data analysis was conducted using the ['Metacells' pipeline](https://github.com/tanaylab/metacells) in R, resulting in the generation of a metadata file that includes information on cell types, embryo origin, and treatment specifics. Additionally, a gene expression matrix detailing the expression levels of genes across cells (genes x cells) was derived from this analysis.

## Technical Details

### Running the Program

To execute the program, use:

```python
python .\bmp_response.py
```

### User Input

Upon running the program, the graphical user interface (GUI) prompts users for:

1. `Fold change`: Controls the threshold for fold change of gene expression.
2. `Minimal expression`: Sets the minimum expression level for genes to be included
3. `Minimal border`:  Minimal value for the heatmap color scale.
4. `Maximal border`:  Maximal value for the heatmap color scale.
5. `Regulation`: genes that show Up and Downregulated, Upregulated, Downregulated in respect to control.
6. `Save heatmap`: saves a .jpeg of the current heatmap display.
7. `Generate PCA`: generate a PCA of the wildtype atlas over the current genes display and saves a .jpeg.

* all explained within the GUI for convinience

### Output

The output is an interactive GUI featuring:

* Heatmap visualization of differentially expressed genes in response to BMP4.
* Optional saving of the heatmap as a .jpeg
* Optional saving of the PCA plot as a .jpeg - the PCA will represent the PC distribution over the selecter genes.

### Dependencies

The program automatically installs all required packages listed in `requirements.txt`. If any packages are missing from your environment, the program will prompt for installation during execution.

### Testing

Test the program using `pytest`.

## Acknowledgment

This project was originally implemented as part of the [Python programming course](https://github.com/szabgab/wis-python-course-2024-04) at the [Weizmann Institute of Science](https://www.weizmann.ac.il/) taught by [Gabor Szabo](https://szabgab.com/).
