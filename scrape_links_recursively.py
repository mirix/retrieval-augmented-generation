### SQE WEB SCRAPER ###
# Extract text from all the static HTML content and save it as txt
# Download all PDFs

import os
import shutil
import requests
import pandas as pd
from bs4 import UnicodeDammit
from bs4 import BeautifulSoup
from inscriptis import get_text

# Download folder
pdfs = 'pdfs'
if os.path.exists(pdfs):
	shutil.rmtree(pdfs)
os.makedirs(pdfs)

# Acceptable URLs must start with html
http = 'http'
# Starting page for scrapping
base_url = 'https://en.swissquote.lu'
# Allowed domains:
# Only en.swissquote.lu (all English content)
# PDFs may be stored in library.swissquote.com or links.imagerelay.com
allowed_domains = ['en.swissquote.lu', 'library.swissquote.com', 'links.imagerelay.com']

# Function to obtain a list of hyperlinks from each page
def get_links(url):
	response = requests.get(url)
	content_type = response.headers.get('content-type')
	
	link_list = []
	
	# Do not extract links from PDFs
	#if 'text/html' in content_type:	
	if not 'application/pdf' in content_type:
				
		soup = BeautifulSoup(UnicodeDammit(response.content).unicode_markup, 'html.parser')
		
		links = soup.find_all('a')
		
		# For each link
		for link in links:
			# Clean up the URL
			link = str(link.get('href')).replace('http://', 'https://').rstrip('/').split('#')[0].split('?')[0].strip()
			# Check if it starts with http
			if link.startswith(http):
				# Check if it is an allowed domain
				if link.split('/')[2] in allowed_domains:
					# Check that is not already in the list
					if link not in link_list:
						link_list.append(link)
		
	return link_list

# Apply the link retrieval function to the first page	
first_links = get_links(base_url)

old_links = first_links
search_links = first_links
full_links = first_links

new_count = len(first_links)
print(new_count)

# Extract links recursively from all pages
i = 1
count_list = [0, new_count]
# Until there are no new links
while count_list[i] > count_list[i-1]:
	
	new_links = []
	for url in search_links:
		temp_links = get_links(url)
		new_links.extend(temp_links)	
	new_links = list(set(new_links))
	
	full_links = list(set(old_links + new_links))
	new_count = len(full_links)
	
	count_list.append(new_count)
	
	search_links = list(set(new_links) - set(old_links))
	
	old_links = full_links
	
	print(len(full_links))
	
	i = i + 1

# Create dataframe with all links
df = pd.DataFrame(full_links, columns=['url'])
df = df.sort_values(by=['url'], ascending=True)

df = df.reset_index()
print(df)

# If the link is a PDF, download it
# It the link is HTML, extract the text
def html_to_pdf(row):
	
	index = row['index']
	url = row['url']
	
	base_name = url.rstrip('/').split('/')[-1].replace('-', '_')
	
	response = requests.get(url)
	content_type = response.headers.get('content-type')
	
	if 'application/pdf' in content_type:
		file_name = base_name + '_' + str(index).zfill(4) + '.pdf'
		open(pdfs + '/' + file_name, 'wb').write(response.content)
	else:
		file_name = base_name + '_' + str(index).zfill(4) + '.txt'
		open(pdfs + '/' + file_name, 'w').write(get_text(response.text))
	
	return file_name

df['file_name'] = df.apply(lambda row: html_to_pdf(row), axis=1)

df.to_csv('sqe_all_links.csv', encoding='utf-8', index=False)
print(df)

