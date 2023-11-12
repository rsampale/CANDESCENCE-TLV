import matplotlib.pyplot as plt # plotting library
import numpy as np
import torch
import pandas as pd
import re
from PIL import Image
import matplotlib.offsetbox as offsetbox
from sklearn.preprocessing import OneHotEncoder
from scipy.ndimage import zoom
import os
import seaborn as sns

def reconstruction_compare(arguments, dataset, encoder, decoder, n = 10): # by default show 10 comparisons

   plt.figure(figsize=(16,4.5)) # CHANGE FIGURE SIZE?
    
    # Randomly choose n indices from dataset for visualization
   random_indices = np.random.choice(len(dataset), size=n, replace=False)
    
   for i, idx in enumerate(random_indices):
      ax = plt.subplot(2, n, i+1)
      img = dataset[idx][0].unsqueeze(0).to(arguments['DEVICE'])
      OH_tensor = dataset[idx][1]
      OH_tensor = OH_tensor.to(arguments['DEVICE'])
      # print(OH_tensor)
      # print(img.shape)
      # OH_tensor = OH_tensor.unsqueeze(-1) # Comment this out omce using variables again
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img, OH_tensor), OH_tensor) ## ADD INPUT FOR OH TENSOR
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')

   # Save the figure, it will overwrite figs with the same name, so only the last figure/epoch of the experiment is really saved:
   file_name = arguments['exp'] + "_reconstruct-compare.jpg"
   plt.savefig(os.path.join(arguments['GRAPH_SAVES'], file_name))
   
   plt.show()

# def get_latent_vectors(dataset, indices, encoder, arguments):
#    image_filenames = dataset.get_image_filename(indices)
#    latent_vectors = []
#    all_conditional_info = [[get_plate_id(path), get_growth_medium(path), get_day(path)] for path in image_filenames] ## change: DONE
#    # OH_encoder = get_one_hot_encoder(all_conditional_info)
#    OH_encoder = dataset.get_dataset_OH_encoder()

#    with torch.no_grad():
#       for i, img in enumerate(image_filenames):
#             image = Image.open(img)

#             if dataset.transform is not None:
#                 image = dataset.transform(image)

#             image = image.to(arguments['DEVICE'])
#             image = image.unsqueeze(0)

#             indiv_conditional_info = all_conditional_info[i]
#             indiv_cond_info_as_string = [f"{indiv_conditional_info[0]}_{indiv_conditional_info[1]}_{indiv_conditional_info[2]}"]
#             indiv_cond_info_string_2d = np.array(indiv_cond_info_as_string).reshape(-1, 1)
#             indiv_OH_encoding = OH_encoder.transform(indiv_cond_info_string_2d)
#             OH_encoding_as_tensor = torch.from_numpy(indiv_OH_encoding)
#             OH_encoding_as_tensor = OH_encoding_as_tensor.to(arguments['DEVICE'])  # Contains image ID as a one-hot encoding
#             size_OH = get_size_onehot_from_tensor(image, arguments['size_bins'])  # gets image size as a one-hot encoding
#             size_OH = size_OH.to(arguments['DEVICE'])
#             # print(f"concatenating: {OH_encoding_as_tensor.shape} + {size_OH.shape}")
#             OH_encoding_as_tensor = torch.cat((OH_encoding_as_tensor, size_OH),dim=1)  # joins the two one-hot encodings into a single vector
#             latent_vector = encoder.encode(image, OH_encoding_as_tensor)  
#             latent_vector = latent_vector.squeeze().cpu().numpy()

#             latent_vectors.append(latent_vector)

#    df = pd.DataFrame(latent_vectors, columns=[f'V{i+1}' for i in range(len(latent_vectors[0]))])
#    df['file_name'] = image_filenames

#    return df

def get_latent_vectors(dataset, indices, encoder, arguments):
   image_filenames = dataset.get_image_filename(indices)
   latent_vectors = []
   image_OH_pairs = [dataset[i] for i in indices]

   with torch.no_grad():
      for img, OH_tensor in image_OH_pairs:
            img = img.to(arguments['DEVICE'])
            OH_tensor = OH_tensor.to(arguments['DEVICE'])
            # OH_tensor = OH_tensor.unsqueeze(-1) # remove when using variables
            latent_vector = encoder.encode(img.unsqueeze(0), OH_tensor)  
            latent_vector = latent_vector.squeeze().cpu().numpy()

            latent_vectors.append(latent_vector)

   df = pd.DataFrame(latent_vectors, columns=[f'V{i+1}' for i in range(len(latent_vectors[0]))])
   df['file_name'] = image_filenames

   return df

def dataframe_w_latentvecs(arguments, dataset, indices, model):
   file_dir = arguments['METADATA_DIR']
   meta_table = pd.read_csv(file_dir, sep='\t',encoding='ISO-8859-1')
   vectors_w_files = get_latent_vectors(dataset, indices, model.encoder, arguments)

   # Extract plate number
   plate_numbers = vectors_w_files['file_name'].str.extract('Pl(\d+)|P(\d+)|Pwt', expand=False)
   vectors_w_files['Plate'] = np.where(plate_numbers[0].notnull(), plate_numbers[0], plate_numbers[1])
   vectors_w_files['Plate'] = vectors_w_files['Plate'].fillna(-1).astype(int)

   # Extract medium
   vectors_w_files['medium'] = vectors_w_files['file_name'].str.extract('(spider|ctrl|spdr|control|serum|RPMI|YPD)', expand=False)
   vectors_w_files['medium'] = vectors_w_files['medium'].replace({'spdr': 'spider', 'ctrl': 'control'})

   # Extract day
   vectors_w_files['day'] = vectors_w_files['file_name'].str.extract('day(\d+)', expand=False).astype(int)

   # Extract replicate
   vectors_w_files['replicate'] = vectors_w_files['file_name'].str.extract(r'_(\d+)-', expand=False).fillna('-1')
   vectors_w_files['replicate'] = vectors_w_files['replicate'].astype(int)

   # Extract position
   vectors_w_files['Position'] = vectors_w_files['file_name'].str.extract('-(.*)\\.', expand=False)

   # Create a function to format position
   def get_row_col(pos):
      # print(pos)
      m = re.search(r"r(\d+)-c(\d+)", pos)
      # print(m)
      if m is not None:
         row = chr(int(m.group(1)) + 64)  # converting to ASCII
         col = m.group(2)
         return row + col
      else:
         return pos

   vectors_w_files['Position'] = vectors_w_files['Position'].apply(get_row_col)

   # Merge the dataframes
   MASTER = pd.merge(vectors_w_files, meta_table, on=["Plate", "Position"], how='inner')

   return MASTER

def create_img_scatterplot(df, arguments, reduction_technique):

   dim1 = ''
   dim2 = ''
   if reduction_technique.lower() == 'tsne':
      dim1 = 'tsne-2d-one'
      dim2 = 'tsne-2d-two'
   elif reduction_technique.lower() == 'umap':
      dim1 = 'umap-2d-one'
      dim2 = 'umap-2d-two'
   else:
      raise ValueError("Please specify a propert dimension reduction technique. Aborting craete_img_scatterplot().")

   # Define the figure and axis
   fig, ax = plt.subplots(1, figsize=(12, 9))

   # Loop over each row and plot the image
   for i, row in df.iterrows():
         img = Image.open(row['file_name'])
         img.thumbnail((50, 50))  # Set the thumbnail size
         img = np.array(img)
        
         # Use OffsetImage and AnnotationBbox to place the image on the plot
         img = offsetbox.OffsetImage(img, zoom=0.5)  # Adjust zoom level as needed
         img_box = offsetbox.AnnotationBbox(img, (row[dim1], row[dim2]),
                                           box_alignment=(0,0),
                                           pad=0,
                                           frameon=False)
         ax.add_artist(img_box)
    
   # Configure the axes
   ax.set_xlim(df[dim1].min(), df[dim2].max())
   ax.set_ylim(df[dim1].min(), df[dim2].max())

   # Save the plot to the saved graph directory
   file_name = arguments['exp'] + "_" + reduction_technique + "_img-scatter.jpg"
   fig.savefig(os.path.join(arguments['GRAPH_SAVES'], file_name))

   # Display the plot
   plt.show()

def create_model_view_img_scatterplot(df, dataset, arguments): ### Doesn't work properly, images out of order

    # Define the figure and axis
   fig, ax = plt.subplots(1, figsize=(12, 9))

   # Loop over each row and plot the image
   for i, row in df.iterrows():
      img_astensor = dataset[i][0].unsqueeze(0).to(arguments['DEVICE'])
      # turn img into an actual displayable image, from the tensor representation present in dataset
      img = img_astensor.cpu().squeeze().numpy()
      
      y_zoom_factor = 50 / img.shape[0]
      x_zoom_factor = 50 / img.shape[1]
      img = zoom(img, (y_zoom_factor, x_zoom_factor))

      # Use OffsetImage and AnnotationBbox to place the image on the plot
      img = offsetbox.OffsetImage(img, zoom=0.5)  # Adjust zoom level as needed
      img_box = offsetbox.AnnotationBbox(img, (row['tsne-2d-one'], row['tsne-2d-two']),
                                          box_alignment=(0,0),
                                          pad=0,
                                          frameon=False)
      ax.add_artist(img_box)
   
   # Configure the axes
   ax.set_xlim(df['tsne-2d-one'].min(), df['tsne-2d-one'].max())
   ax.set_ylim(df['tsne-2d-two'].min(), df['tsne-2d-two'].max())

   # Display the plot
   plt.show()

def create_annotated_scatterplot(df, anno_type):
   # Define the figure and axis
   fig, ax = plt.subplots(1, figsize=(12, 9))

   # Create a color palette that includes all unique values in anno_type
   palette = sns.color_palette('tab10', n_colors=len(df[anno_type].unique()))
   # Convert the palette to a dictionary
   color_dict = dict(zip(df[anno_type].unique(), palette))

   # Set 'nan', 'NaN', and 'unknown' to grey
   for key in ['nan', 'NaN', 'unknown']:
      if key in color_dict:
         color_dict[key] = (0.5, 0.5, 0.5)  # RGB for grey

   # Create the scatterplot, color points by an annotation type like 'geography_general'
   sns.scatterplot(data=df, x='tsne-2d-one', y='tsne-2d-two', hue=anno_type, ax=ax, palette='tab10')
   
   # Configure legend
   legend_labels = {key: 'Unknown' if key in ['nan', 'NaN', 'unknown'] else key for key in df[anno_type].unique()}
   for t, l in zip(ax.get_legend().texts, ax.get_legend().get_lines()):
        t.set_text(legend_labels[t.get_text()])

   # Configure the axes
   ax.set_xlim(df['tsne-2d-one'].min(), df['tsne-2d-one'].max())
   ax.set_ylim(df['tsne-2d-two'].min(), df['tsne-2d-two'].max())

   # Display the plot
   plt.show()
   print("end")

def get_plate_id(image_path):
   match = re.search(r'Pl(\d+)|P(\d+)|(Pwt)', image_path)
   if match:
      if match.group(1) is not None:
          return int(match.group(1))
      elif match.group(2) is not None:
          return int(match.group(2))
      elif match.group(3) is not None:
          return -1  # Or whatever value you want to use for 'Pwt'
   else:
      return -1

def zoom_img_scatterplot(df, arguments, reduction_technique, zoom_region, annotate=None):

   dim1 = ''
   dim2 = ''
   if reduction_technique.lower() == 'tsne':
      dim1 = 'tsne-2d-one'
      dim2 = 'tsne-2d-two'
   elif reduction_technique.lower() == 'umap':
      dim1 = 'umap-2d-one'
      dim2 = 'umap-2d-two'
   else:
      raise ValueError("Please specify a proper dimension reduction technique. Aborting craete_img_scatterplot().")

   # Define the figure and axis
   fig, ax = plt.subplots(1, figsize=(12, 9))

   if annotate is None:
      # Loop over each row and plot the image
      for i, row in df.iterrows():
            img = Image.open(row['file_name'])
            img.thumbnail((60, 60))  # Set the thumbnail size larger
            img = np.array(img)

            # Use OffsetImage and AnnotationBbox to place the image on the plot
            img = offsetbox.OffsetImage(img, zoom=0.6)  # Adjust zoom level as needed
            img_box = offsetbox.AnnotationBbox(img, (row[dim1], row[dim2]),
                                               box_alignment=(0,0),
                                               pad=0,
                                               frameon=False)
            ax.add_artist(img_box)
   else:
      # Create a color palette that includes all unique values in anno_type
      palette = sns.color_palette('tab10', n_colors=len(df[annotate].unique()))
      # Convert the palette to a dictionary
      color_dict = dict(zip(df[annotate].unique(), palette))

       # Set 'nan', 'NaN', and 'unknown' to grey
      for key in ['nan', 'NaN', 'unknown']:
         if key in color_dict:
            color_dict[key] = (0.5, 0.5, 0.5)  # RGB for grey

       # Create the scatterplot, color points by an annotation type like 'geography_general'
      sns.scatterplot(data=df, x=dim1, y=dim2, hue=annotate, ax=ax, palette=color_dict)

      # Configure legend
      legend_labels = {key: 'Unknown' if key in ['nan', 'NaN', 'unknown'] else key for key in df[annotate].unique()}
      for t, l in zip(ax.get_legend().texts, ax.get_legend().get_lines()):
            t.set_text(legend_labels[t.get_text()])

   # Configure the axes
   xmin, xmax, ymin, ymax = zoom_region
   ax.set_xlim(xmin, xmax)
   ax.set_ylim(ymin, ymax)

   # Save the plot to the saved graph directory
   file_name = arguments['exp'] + "_" + reduction_technique + ("_zoom-annotate-scatter.jpg" if annotate else "_zoom-scatter.jpg")
   fig.savefig(os.path.join(arguments['GRAPH_SAVES'], file_name))

   # Display the plot
   plt.show()

def get_growth_medium(image_path):
    match = re.search('(spider|ctrl|spdr|control|serum|RPMI|YPD)', image_path)
    if match is not None:
        medium = match.group(0)
        medium = medium.replace('spdr', 'spider').replace('ctrl', 'control')
        return medium
    else:
        return None
    
def get_day(image_path):
   match = re.search(r'day(\d+)', image_path)
   if match:
      return int(match.group(1))
   else:
      return -1

def get_one_hot_encoder(categorical_info): # takes as input a list of lists, inner lists are category values
   one_hot_encoder = OneHotEncoder(sparse_output=False)

   categorical_info_as_strings = [f"{info[0]}_{info[1]}_{info[2]}" for info in categorical_info]
   categorical_info_as_strings_2d = np.array(categorical_info_as_strings).reshape(-1, 1)

#    print(plate_ids_and_mediums)
   one_hot_encoder.fit(categorical_info_as_strings_2d) # takes categorical info in 2d string format, fits one hot encoder to that info

   return one_hot_encoder

def get_size_onehot_from_tensor(img_tensor, bins):
   # Shape if input comes from VAE call is ([48, 1, 135, 135])
   device = img_tensor.device
   thresh = 0.5
   binary_image = torch.where(img_tensor > thresh, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
   num_ones = torch.sum(binary_image)
   num_pixels = binary_image.numel()
   proportion_of_ones = num_ones.float() / num_pixels
   # print(f"shape of proportions_of_ones: {proportions_of_ones.shape}")
   # print(f"first element prop: {proportions_of_ones[10]}")

   # Go through the tensor of proportions, assigning a bucket ID instead
   buckets = np.linspace(0, 1, bins+1)[1:] ## divide into buckets according to how many bins we want
   
   # fit the one-hot encoder to the size ids
   ids_for_fit = [[i] for i in range(1,bins+1)]
   encoder = OneHotEncoder(sparse_output=False)
   trained_oh_encoder = encoder.fit(ids_for_fit)

   id = 1
   for bucket in buckets:
      if proportion_of_ones.item() <= bucket:
         break
      id += 1
   proportion_as_id = [[id]]

   one_hot_size_id = trained_oh_encoder.transform(proportion_as_id)
   # print(f"proportions_as_ids before transform: {proportions_as_ids}")
   # print(f"one_hot_size_ids to be returned: {one_hot_size_ids}")
   one_hot_size_id = torch.from_numpy(one_hot_size_id)
   return one_hot_size_id

def get_intensity_onehot_from_tensor(img_tensor, intensity_bins):
   # Compute the average intensity of the image
    avg_intensity = img_tensor.mean().item()
    
    # Create intensity bucket boundaries
    buckets = np.linspace(0, 1, intensity_bins+1)[1:] 
   
    # Assign the image to its corresponding bucket
    id = 1
    for bucket in buckets:
        if avg_intensity <= bucket:
            break
        id += 1
        
    # Prepare the one-hot encoder
    ids_for_fit = np.arange(1, intensity_bins+1).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(ids_for_fit)

    # Transform the intensity bucket ID to its one-hot encoded form
    one_hot_intensity_id = encoder.transform(np.array([[id]]))
    one_hot_intensity_id = torch.from_numpy(one_hot_intensity_id)

    return one_hot_intensity_id


def create_loss_graph(train_loss, val_loss, arguments):
   plt.figure(figsize=(10,5))
   plt.plot(train_loss, label='Training loss')
   plt.plot(val_loss, label='Validation loss')

   plt.yscale('log')

   # labels
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('VAE Loss over Epochs')

   # legend
   plt.legend()

   file_name = arguments['exp'] + "_loss-graph.jpg"
   plt.savefig(os.path.join(arguments['LOSS_SAVES'], file_name))

   plt.show()

