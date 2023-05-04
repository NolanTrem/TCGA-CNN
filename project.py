#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/NolanTrem/TCGA-CNN/blob/main/project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import pandas as pd
url = 'https://raw.githubusercontent.com/NolanTrem/HE2RNA_code/master/metadata/samples_description.csv'
url_manifest = "https://raw.githubusercontent.com/NolanTrem/HE2RNA_code/master/gdc_manifests/gdc_manifest.2018-03-13_alltranscriptome.txt"
transcriptome = pd.read_csv(url, delimiter='\t')
transcriptome_manifest_prep = pd.read_csv(url_manifest, delimiter='\t')


# In[2]:


#brain cancer files
url2 = "https://raw.githubusercontent.com/NolanTrem/TCGA-CNN/main/manifests/fullManifests/brainManifestFull.txt"
brain = pd.read_csv(url2, delimiter='\t', skiprows=1, names=['id', 'filename', 'md5', 'size', 'state', 'tumor'])

#breast cancer files
url3 = "https://raw.githubusercontent.com/NolanTrem/TCGA-CNN/main/manifests/fullManifests/breastManifestFull.txt"
breast = pd.read_csv(url3, delimiter='\t', skiprows=1, names=['id', 'filename', 'md5', 'size', 'state', 'tumor'])

#lung cancer files
url4 = "https://raw.githubusercontent.com/NolanTrem/TCGA-CNN/main/manifests/fullManifests/bronchusAndLungManifestFull.txt"
lung = pd.read_csv(url4, delimiter='\t', skiprows=1, names=['id', 'filename', 'md5', 'size', 'state', 'tumor'])

#kidney cancer files
url5 = "https://raw.githubusercontent.com/NolanTrem/TCGA-CNN/main/manifests/fullManifests/kidneyManifestFull.txt"
kidney = pd.read_csv(url5, delimiter='\t', skiprows=1, names=['id', 'filename', 'md5', 'size', 'state', 'tumor'])

#ovary cancer files
url6 = "https://raw.githubusercontent.com/NolanTrem/TCGA-CNN/main/manifests/fullManifests/ovarianManifestFull.txt"
ovary = pd.read_csv(url5, delimiter='\t', skiprows=1, names=['id', 'filename', 'md5', 'size', 'state', 'tumor'])

merged_cancer = pd.concat([brain, breast, lung, kidney, ovary], axis=0)


# In[3]:


def merge_data(transcriptome, svs):
  svs['case_id'] = svs['filename'].str[:12]
  svs['sample_id'] = svs['filename'].str[:16]
  transcriptome = transcriptome.rename(columns={'Sample.ID': 'sample_id'})

  common_ids = transcriptome[transcriptome['sample_id'].isin(svs['sample_id'])]['sample_id']
  common_ids_df = pd.merge(transcriptome[transcriptome['sample_id'].isin(svs['sample_id'])], svs, on='sample_id')

  common_ids_df = common_ids_df.drop(['Data.Category', 'Data.Type', 'md5', 'Sample.Type', 'size', 'state', 'case_id', 'Case.ID'], axis=1)
  common_ids_df = common_ids_df.rename(columns={'File.ID': 'transcriptome_id', 'File.Name' : 'transcriptome_filename', 
                                                'Project.ID' : 'project_id', 'id' : 'slide_id', 'filename': 'slide_filename'})
  
  return common_ids_df


# First 60 for each type. Exclude brain (995 tumors, 5 solid tissue normal)

# In[4]:


brain1 = merge_data(transcriptome, brain)
breast1 = merge_data(transcriptome, breast)
lung1 = merge_data(transcriptome, lung)
kidney1 = merge_data(transcriptome, kidney)
ovary1 = merge_data(transcriptome, ovary)
brain1['type'] = 'brain'
breast1['type'] = 'breast'
lung1['type'] = 'lung'
kidney1['type'] = 'kidney'
ovary1['type'] = 'ovary'

dfs = [breast1, lung1, kidney1, ovary1]

# create an empty list to hold the selected dataframes
selected_dfs = []

# loop through the dataframes and select the first 60 rows of each label type
for df in dfs:
    primary_tumor = df[df['tumor'] == 'Primary Tumor'].head(60)
    solid_tissue_normal = df[df['tumor'] == 'Solid Tissue Normal'].head(60)
    selected_df = pd.concat([primary_tumor, solid_tissue_normal])
    selected_dfs.append(selected_df)

# concatenate the selected dataframes into a single dataframe
merged_df = pd.concat(selected_dfs, axis=0)
#merged_df = pd.concat(dfs, axis = 0)


# In[5]:


merged_df.to_csv('merged_df.txt', sep='\t', index=False)


# In[ ]:


print(brain[brain['tumor'] == 'Primary Tumor'].shape[0])
print(brain[brain['tumor'] == 'Solid Tissue Normal'].shape[0])


# In[ ]:


transcriptome_manifest = transcriptome_manifest_prep[transcriptome_manifest_prep['filename'].isin(merged_df['transcriptome_filename'])]
wsi_manifest = merged_cancer[merged_cancer['filename'].isin(merged_df['slide_filename'])]

check_manifest = merged_df[merged_df['transcriptome_filename'].isin(transcriptome_manifest_prep['filename'])]

transcriptome_manifest.to_csv('transcriptome_manifest_20230421.txt', sep='\t', index=False)
wsi_manifest.to_csv('wsi_manifest_20230421.txt', sep='\t', index=False)


# In[ ]:


transcriptome_manifest['id'].count()


# In[ ]:


print(len(transcriptome_manifest))


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install gdc-client')

