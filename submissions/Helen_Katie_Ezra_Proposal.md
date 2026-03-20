## Abstract

Torrential rainfall in Nairobi led to flooding with flood waters reaching peak levels on the 7th March, 2026 - [leaving 66 people](https://www.bbc.com/news/articles/cz6e5eng03yo) recorded dead and destruction of property and infrastructure. There full effect of this is yet to be quantified but more importantly a flood risk exposure with satellite imagery and deep learning techniques can aid early warning systems as well as urban planning systems to hedge against any such future flood risks.

## Problem Definition & Use Case

Nairobi, a city of about 5.8 million inhabitants is one of Africa's fastest growing cities. Change in rainfall patterns has recently led to much higher rains than the city often receives. Floods have left many homeless and caused huge property loss and loss of life and livelihood; The empirical evidence of extent and severity of this and many floods is sparse and disparate. This work aims to fill this knowledge gap by;

1. Mapping out the flooded areas with empirical satellite data on area (extent), water depth or volume.
2. Overlay population data to provide a realistic number of how many people have been affected - in such emergency situations it is not uncommon for many cases to go unreported.
3. Propose potential city planning techniques that might better hedge the city from such flood risk by investigating more optimal or efficient use of scarce resources to manage flood risk

We further narrow our study region to Nairobi City given its relevance to the central business district.

![Population Density Heatmap](/submissions/population.jpg)

Map of study area with Population density heatmap

## Technical Justification

68% of the world population is projected to live in urban areas by 2050. Infrastructure and early
warning systems have got to grow in a similar manner in order to withstand this growth in population.

Inadequate infrastructure and systems leave populations exposed to extreme weather events such as flush floods following torrential rains. Rapid forest cutting and environmental degradation renders the land bare and destroys wind breakers such as trees, following the onset of torrential rains, bare soils cave to the intense rain causing flooding. Swamp or marshland reclamation for example has been shown to clog drainage channels or water sinks with flooding as a result of this. This work proposes to train an image segmentation model. We’ll identify a baseline within the study area. 
The model will mostly be looking for two main features;

1. Water where it’s not expected - that is floods.
2. Change in land use from river bank swamp or water channel to built environment.

Our choice is a segmentation model - as relates to floods. We intend to employ a pretrained model for this task.

This is tied to the objective of quantifying the scope and scale of environmental degradation. To the contrary, a classification based approach for example would identify what is a flood - this will provide little useful information. Regression based approaches would be useful if we sought to perform future forecasts - which is not in line with what we seek to achieve.

![Flood Image](/submissions/flood2.png)
Nairobi Floods March 2026

## Challenges 

Equatorial regions are known to often have cloud cover. The phenomenon we intend to study such as floods follow rainfall which makes it most likely an event where cloud cover might pose a challenge. In addition to clouds, shadows too can lead to a lot of false positives. It will be an important task in this study to inspect the images for these dynamic artifacts prior to model training.

Secondly flood waters are transient - in some of these areas there might be flush floods which come and go fast leaving disaster in their wake. If our model only looks for water -in place- where it ought not to be, this may be a clear miss for transient floods such as those when rivers break their banks -which is partly the case in Nairobi.

Thirdly, finding pretrained land use change models on Nairobi might be problematic let alone the rarity of training data for this region.
 
In addition to the above and more related to remote sensing studies in the global south;

There are a couple of challenges based on the region and the time varying nature of what we intend to study. We briefly discuss these below:

1. Lack of availability of labelled data. This issue is not just unique to Nairobi - it’s mostly an issue for the cities in the global south.
2. Availability of satellite imagery for risks as floods is not readily available for the time frame in which the phenomenon can be best studied. And when it is available there are cross cutting issues of lack of very fine enough resolution. It might be possible to combine images from different satellites but that could raise alignment issues as well as differing resolution issues.

## Methodological Precedent

The use of remote sensing and machine learning (ML) and deep learning techniques to identify flooding has a lot of precedent. Bentivoglio and colleagues provide a literature review of papers using deep learning techniques to detect floods, although not necessarily limited to remote sensing data. They provide three useful categories of problems researchers are trying to solve in relation to a flooding event; our project proposal would best be characterized as a flood inundation analysis of a flooding event. A successful analysis can then be used for “post-flood evacuation, protection planning, and for damage assessment. These maps can then also be used as calibration data for other applications such as flood susceptibility or flood hazard mapping.” (pg. 4347). In regards to remote sensing raster data, a common method to determine the flood inundation extent is to treat it as an image segmentation problem, where the output should identify pixels as either flooded (FL) or nonflooded (NF).

Flood inundation mapping is particularly difficult in urban areas because of the diversity and density of buildings and land uses. Peng and colleagues have found that traditional ML techniques struggle with the task of differentiating the difference between flood waters and buildings because of light reflection. They also identify that there is often a high class imbalance between FL and NF in urban areas, further complicating the task. This study employed a patch similarity convolutional neural network (PSNet) in order to improve the time spent on labeling ground truth data and reduce the dimensionality of the model. Thus, their output is at a lower resolution than many applications, such as hydrology engineering, might require. However, they argue that the trade-off is in having a model that is potentially more responsive to emerging conditions. They found that the DL technique they used outperformed the ML techniques in terms of accuracy and F1-score. Interestingly, they also found that there was marginal model improvement when using a bi-temporal model, which was trained to learn based on the difference between a pre- and post- flooding event, and a uni-temporal model, which only learned from the post-flooding samples. The importance of multi-temporal training is further validation by Li and colleagues, who find significant accuracy gains with a CNN trained on multi-temporal coherence intensity. Thus, there is a growing body of research establishing a precedent of the relatively high accuracy of DL techniques over ML or thresholding techniques in identifying urban flood inundation (see also Nemni et. al., 2020). 

Obtaining accurately labeled training data is a barrier in flood inundation mapping. Bonifilia and colleagues, however, find that hand-labeled data, while ultimately important for validation, is not necessarily the best for training CNN models on flood detection. They make publicly available the labeled SEN1FLOODS11 dataset, which we will use in our project (see section 5). 

## Data Plan and Processing

### Study Area and Time Period 
1. Study Area: Nairobi, Kenya
 - Bounding Box:
  - Longitude: 36.6 – 37.2
  - Latitude: −1.5 – −1.0
 2. Study Time Period
The analysis is conducted using a reference date of March 7, 2026. However, due to the ongoing rainy season and the uncertainty of near-future flood conditions, a historical flood event was selected for reliable analysis.

Specifically, we focus on a documented flood event from November 30, 2023, using data from Maxar Open Data Kenya Flooding as a verified referenc (Hochet-Bodin, 2023; STAC Browser, 2024). For the flood event on November 30, 2023, a pair of pre- and post-event Sentinel-1 SAR images was selected to enable change detection analysis. The pre-event image was acquired on November 27, 2023, while the post-event image was acquired on December 2, 2023. These two observations were chosen as the closest available acquisitions surrounding the flood event, ensuring a short temporal interval of five days. This event-proximate selection minimizes the influence of non-flood-related factors, such as seasonal vegetation dynamics, soil moisture variation, and other landscape changes unrelated to the flood event.By reducing temporal variability, this approach improves the reliability of SAR backscatter change detection and helps isolate flood-induced surface changes from other environmental noise.

### Data 

#### Primary Input

The primary input data for this study will be bi-temporal Sentinel-1 SAR imagery, including:

- Pre-flood Sentinel-1 VV
- Pre-flood Sentinel-1 VH
- Post-flood Sentinel-1 VV
- Post-flood Sentinel-1 VH

These data will be used to compare surface conditions before and after the flood event in order to identify areas that were newly inundated after the heavy rainfall. Using both VV and VH polarization is important because they capture different surface scattering characteristics.

- VV is often more sensitive to open water and general surface backscatter patterns
- VH can provide additional information on vegetation, rough surfaces, and more complex flooded environments.

Combining both channels can therefore improve the model’s ability to distinguish floodwater from other land surfaces in a dense urban environment.

#### Secondary Data Source

Sentinel-2 optical imagery will be used as a secondary data source, alongside high-resolution flood imagery from Maxar Open Data Kenya Flooding, to support the interpretation of land cover and surface conditions where cloud-free scenes are available.

This dataset can assist with visual validation, interpretation of ambiguous areas, and identification of features such as built-up land, vegetation, and water-related surfaces.
However, the availability of well-validated and spatially explicit labeled data for the study area remains limited. As a result, it is uncertain whether these auxiliary datasets will fully address this gap. Nevertheless, this study aims to explore their potential contribution and evaluate their effectiveness in improving model performance.

#### Labeled Data Strategy

##### Public labeled dataset

A public flood-related dataset such as Sen1Floods11 will be used as a source of training data for initial model development. Although Sen1Floods11 contains both Sentinel-1 and Sentinel-2 imagery, this project will use only the Sentinel-1 component in order to remain consistent with the main SAR-based modeling framework. The primary role of this dataset is to help the model learn general water-related and flood-relevant patterns from Sentinel-1 imagery and provide a starting point for flood segmentation. At this stage, the labels are mainly used to train the model to distinguish water from non-water, rather than to explicitly separate event-specific floodwater from permanent water

##### Manually labeled local data

A smaller set of manually labeled Nairobi 2026 flood samples will be created to provide higher-confidence local reference data. These local labels will be used both for limited fine-tuning and for validation, with a portion held out for evaluation so that model performance can be assessed on event-specific and location-specific flood conditions rather than relying only on external datasets. To improve labeling efficiency, local annotations will be created on a set of representative image patches rather than for the entire city. This patch-based strategy is especially useful under limited labeling resources and has been discussed in prior flood-mapping studies.

## Modeling Approach & Evaluation

### Modeling Approach

This project formulates flood detection as a bi-temporal semantic segmentation problem. The model will use pre-flood and post-flood Sentinel-1 SAR imagery, including both VV and VH polarizations, to predict a pixel-level flood extent mask for Nairobi following the November 2023 flood events. The bi-temporal design is intended to capture changes in radar backscatter before and after the storm, allowing the model to identify newly inundated areas rather than simply detecting water-like surfaces.

#### Baseline

As a simple baseline, this project will first implement a threshold-based flood mapping approach using changes in Sentinel-1 backscatter between pre-event and post-event imagery. Specifically, flood candidates will be identified based on differences in VV and VH backscatter values before and after the flood event. This approach provides a transparent and easily interpretable reference against which the performance of deep learning models can be evaluated. The implementation follows established workflows, particularly the recommended practice developed by UN-SPIDER in Google Earth Engine for SAR-based flood mapping (United Nations, 2020).

#### Deep Learning Model

This study will explore the use of pretrained models developed on the Sen1Floods11 dataset as a starting point. Leveraging pretrained weights is expected to improve model convergence and generalization, particularly given the limited availability of high-quality labeled data in the study area.

The primary deep learning approach will evaluate two fully convolutional segmentation architectures inspired by prior SAR-based flood mapping research: U-Net and XNet. In the referenced study (Bonifilia et al), U-Net is described as a standard encoder–decoder architecture with skip connections that support precise spatial localization, while XNet extends this design with a symmetric encoder–decoder–encoder–decoder structure, aiming to better preserve boundary-level detail and small-scale features.

For this project:

- U-Net will serve as the main segmentation model because it is widely used, relatively efficient, and well suited to limited labeled data settings.
- XNet will be explored as a comparative architecture because its multi-decoder design may improve delineation of flood boundaries in complex urban surfaces.

The model inputs will include:

- pre-flood Sentinel-1 VV
- pre-flood Sentinel-1 VH
- post-flood Sentinel-1 VV
- post-flood Sentinel-1 VH

If data preparation allows, additional terrain-related variables such as DEM-derived slope may be concatenated as auxiliary channels to examine whether terrain context improves flood extent segmentation.

### Evaluation 

Model performance will be evaluated using standard segmentation metrics. Following the evaluation framework used in the reference study, the main metrics will include:

- Precision
- Recall
- F1-score / Dice coefficient
- Critical Success Index (CSI)

These metrics capture different aspects of model performance, including detection accuracy, completeness, and the balance between false positives and false negatives.

To assess the added value of deep learning, results from the U-Net and XNet models will be directly compared with the threshold-based baseline. This comparison will focus on identifying key strengths of deep learning approaches, such as improved boundary delineation, reduced noise, and enhanced detection of small or heterogeneous flood patterns.

#### Validation Strategy

Due to the limited availability of high-quality, validated ground-truth data for the study area, a fully quantitative validation is not feasible. Instead, this study adopts a multi-source validation strategy.

First, detected flood extents will be compared with reported impact data, such as spatial patterns of flood-related fatalities or damage, to assess whether identified inundation areas align with known high-impact locations. Second, elevation data (DEM) will be used to evaluate the physical plausibility of the results, under the assumption that flood-prone areas are more likely to occur in low-lying and hydrologically connected regions.

While these approaches do not replace fully labeled validation datasets, they provide a reasonable proxy for assessing the spatial reliability and realism of the model outputs.

## References

Bentivoglio, Roberto, Elvin Isufi, Sebastian Nicolaas Jonkman, and Riccardo Taormina. “Deep Learning Methods for Flood Mapping: A Review of Existing Applications and Future Research Directions.” Hydrology and Earth System Sciences 26, no. 16 (2022): 4345–78. https://doi.org/10.5194/hess-26-4345-2022.

Bonafilia, Derrick, Beth Tellman, Tyler Anderson, and Erica Issenberg. “Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), June 2020, 835–45. https://doi.org/10.1109/CVPRW50498.2020.00113.

Hochet-Bodin, Noé. “Ethiopia, Kenya and Somalia hit by unprecedented floods.” Africa,Climate Change. Le Monde, November 30, 2023. https://www.lemonde.fr/en/le-monde-africa/article/2023/11/30/ethiopia-kenya-and-somalia-hit-by-unprecedented-floods_6300716_124.html.

Nemni, Edoardo, Joseph Bullock, Samir Belabbes, and Lars Bromley. “Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery.” Remote Sensing 12, no. 16 (2020): 2532. https://doi.org/10.3390/rs12162532.

Peng, Bo, Zonglin Meng, Qunying Huang, and Caixia Wang. “Patch Similarity Convolutional Neural Network for Urban Flood Extent Mapping Using Bi-Temporal Satellite Multispectral Imagery.” Remote Sensing 11, no. 21 (2019): 2492. https://doi.org/10.3390/rs11212492.

United Nations. “Step-by-Step: Recommended Practice: Flood Mapping and Damage Assessment Using Sentinel-1 SAR Data in Google Earth Engine | UN-SPIDER Knowledge Portal.” 2020. https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step.

Winter, Jospeh, and Richard Kagoe. “Kenya Floods Leave 66 Dead, as Heavy Rains Continue in Nairobi.” March 15, 2026. https://www.bbc.com/news/articles/cz6e5eng03yo.

STAC Browser. “105001003B003900.” 2024. Accessed March 20, 2026. https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.amazonaws.com/events/Kenya-Flooding-May24/ard/acquisition_collections/105001003B003900_collection.json?.language=en.

STAC Browser. “Maxar-Opendata.” Accessed March 20, 2026. https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.amazonaws.com/events/catalog.json?.language=en.
