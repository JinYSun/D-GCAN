
<h3 align="center">
<p> A deep learning-based process related to the screening of SARS-CoV2 3CL inhibitors.<br></h3>
<h4 align="center">
---

Coronavirus disease 2019 (COVID-19) is a highly infectious disease caused by severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2). It is urgent to find potential antiviral drugs against SARS-CoV-2 in a short time. Deep learning-based virtual screening is one of the approaches that can rapidly search against large molecular libraries. Here, SARS-CoV-2 3C-like protease (SARS-CoV-2 3CLpro) was chosen as the target. As shown in Figure bellow, the utility of D-GCAN is evaluated by comparing the screening results on the GDB-13 and S-GDB13 databases. The process was carried out with the help of the transfer learning method (Wang et al., 2021), DeepPurpose (Huang et al., 2020), and ADMETLab2.0 (Xiong et al., 2021). 

These databases were firstly screened by using a transfer learning method (COVIDVS) proposed by Wang et al (Wang et al., 2021), which was reported for screening inhibitors against SARS-CoV-2. The model was trained on the dataset containing inhibitors against HCoV-OC43, SARS-CoV and MERS-CoV. All of these viruses as well as SARS-CoV-2 belong to β-coronaviruses. They have high consistency in essential functional proteins (Wu et al., 2020; Shen et al.; Pillaiyar et al., 2020). Then, the trained model was fine-tuned by the transfer learning approach with the dataset containing drugs against SARS-CoV-2. In this way, 107 million drug-like molecules were screened out. Then, drug-target interaction prediction (DTI) was carried out based on DeepPurpose (Huang et al., 2020), which provided pretrained model for the interaction prediction between drugs and SARS-CoV-2 3CLpro target. The interaction binding score was evaluated by the dissociation equilibrium constant (Kd). After this step, 17 thousand molecules with high affinity were obtained. Finally, ADMET properties were widely chosen and used for screening SARS-CoV-2 inhibitors (Gajjar et al., 2021; Roy et al., 2021; Dhameliya et al., 2022). These properties were calculated by using ADMETLab2.0 (Xiong et al., 2021), and 65 candidates with good properties were selected.

![图片](https://user-images.githubusercontent.com/62410732/176149139-b96f2edd-b66b-4007-a0f4-73259b319cb6.png)


## COVIDVS

COVIDVS models are Chemprop models trained with anti-beta-coronavirus actives/inactives collected from published papers and fine-tuned with anti-SARS-CoV-2 actives/inactives.



## DeepPurpose

DeepPurpose has provied the pretrained model by predicting the interaction between a target (SARS-CoV2 3CL Protease) and a list of repurposing drugs from a curated drug library of 81 antiviral drugs. The Binding Score is the Kd values. Results aggregated from five pretrained model on BindingDB dataset.



## AMETLab2.0

Undesirable pharmacokinetics and toxicity of candidate compounds are the main reasons for the                    failure of drug development, and it has been widely recognized that absorption, distribution,                    metabolism, excretion and toxicity (ADMET) of chemicals should be evaluated as early as possible.                    ADMETlab 2.0 is an enhanced version of the widely used [ADMETlab](http://admet.scbdd.com/) for systematical evaluation of ADMET properties, as well as some physicochemical properties and medicinal chemistry friendliness. With significant updates to functional modules, predictive models, explanations, and                    the user interface, ADMETlab 2.0 has greater capacity to assist medicinal chemists in accelerating                    the drug research and development process.                



## Acknowledgement

Dhameliya,T.M. *et al.* (2022) Systematic virtual screening in search of SARS CoV-2 inhibitors against spike glycoprotein: pharmacophore screening, molecular docking, ADMET analysis and MD simulations. *Mol Divers*.

Gajjar,N.D. *et al.* (2021) In search of RdRp and Mpro inhibitors against SARS CoV-2: Molecular docking, molecular dynamic simulations and ADMET analysis. *Journal of Molecular Structure*, **1239**, 130488.

Huang,K. *et al.* (2020) DeepPurpose: a deep learning library for drug–target interaction prediction. *Bioinformatics*, **36**, 5545–5547.

Pillaiyar,T. *et al.* (2020) Recent discovery and development of inhibitors targeting coronaviruses. *Drug Discovery Today*, **25**, 668–688.

Roy,R. *et al.* (2021) Finding potent inhibitors against SARS-CoV-2 main protease through virtual screening, ADMET, and molecular dynamics simulation studies. *Journal of Biomolecular Structure and Dynamics*, **0**, 1–13.

Shen,L. *et al.* High-Throughput Screening and Identification of Potent Broad-Spectrum Inhibitors of Coronaviruses. *Journal of Virology*, **93**, e00023-19.

Wang,S. *et al.* (2021) A transferable deep learning approach to fast screen potential antiviral drugs against SARS-CoV-2. *Briefings in Bioinformatics*.

Wu,F. *et al.* (2020) A new coronavirus associated with human respiratory disease in China. *Nature*, **579**, 265–269.

Xiong,G. *et al.* (2021) ADMETlab 2.0: an integrated online platform for accurate and comprehensive predictions of ADMET properties. *Nucleic Acids Research*, **49**, W5–W14.



