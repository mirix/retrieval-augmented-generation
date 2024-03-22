import warnings
warnings.filterwarnings('ignore')

import os
n_cores = os.cpu_count()//2
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# Adjust according to the number of GPUs you wish to use
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import time as t
import re
from datetime import datetime
from fpdf import FPDF
from pypdf import PdfWriter
from unidecode import unidecode
import plotly.graph_objects as go
#import plotly.express as px

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import set_seed
set_seed(0)

max_tokens = 1536

model_name = 'MaziyarPanahi/samantha-1.1-westlake-7b-AWQ'
#model_name = 'TheBloke/dolphin-2.6-mistral-7B-dpo-laser-AWQ'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', low_cpu_mem_usage=True)
# device_map='cuda:0'								
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, max_length=1536, truncation=True)

### VARIABLES ###

now = datetime.now()
today = now.strftime('%d_%m_%Y')
hour = now.strftime('%H:%M')

# Number of results
#n_res = 10
# Where to save the pdf with the search results
down_path = '/tmp'
pdf_name = 'search.pdf'
pdf_path = down_path + '/' + pdf_name

### WEBDRIVER OPTIONS ###

options = webdriver.FirefoxOptions()
# Proxy (only if connection is tunnelled)
options.set_preference('network.proxy.type', 1)
options.set_preference('network.proxy.socks', '127.0.0.1')
options.set_preference('network.proxy.socks_port', 1337)
options.set_preference('network.proxy.socks_version', 5)
options.set_preference('profile', '/home/emoman/.mozilla/firefox/abcw93dn.default')

# Try to conceal the bot and appear as a human
# (only works in Chrome, so useless here)
options.add_argument("--disable-blink-features=AutomationControlled") 

# Browser agent (how does the browser indentify itself
options.add_argument
("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
+"AppleWebKit/537.36 (KHTML, like Gecko)"
+"Chrome/87.0.4280.141 Safari/537.36")

# Do not open GUI
options.add_argument('--headless')

# Try to load a site as tidy as possible
options.add_argument("--disable-infobars")
options.add_argument("--disable-notifications")
options.add_argument("--disable-popup-blocking")

# We have added extensions to accept cookies 
# as well as to remove adds, so we want them up and running
#options.add_argument("--disable-extensions")

# Print options
options.set_preference("print.always_print_silent", True)
options.set_preference("print.show_print_progress", False)
options.set_preference('print.save_as_pdf.links.enabled', True)
options.set_preference("print.print_headerleft", "")
options.set_preference("print.print_headerright", "")
options.set_preference("print.print_footerleft", "")
options.set_preference("print.print_footerright", "")
options.set_preference("browser.download.folderList", 2)
options.set_preference("browser.download.dir", down_path)
options.set_preference("browser.download.useDownloadDir", True)
options.set_preference("pdfjs.disabled", True)
options.set_preference("print_printer", "Mozilla Save to PDF")
options.set_preference("print.printer_Mozilla_Save_to_PDF.print_to_file", True)
#options.set_preference('print.printer_Mozilla_Save_to_PDF.print_to_filename', pdf_path)
options.set_preference('print.printer_Mozilla_Save_to_PDF.print_to_filename', '{}'.format(pdf_path))
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")

# Use pre-existing Firefox profile
# (in order to prevent Google's agreement from and make sure English is the default language)
profile_path = '/home/emoman/.mozilla/firefox/abcw93dn.default'
options.add_argument('-profile')
options.add_argument(profile_path)

# INITIALISE WEBDRIVER (geckodriver = Firefox)
driver = webdriver.Firefox(options=options)
# Useless unless we want a screenshot (as opposed to a PDF)
driver.maximize_window()
# Try to conceal the bot and appear as a human (probably, not working)
driver.execute_script("return navigator.userAgent")
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})") 

### MODEL ###

generation_params = {
    'do_sample': True,
    'temperature': 0.0000001,
    'top_p': 0.0000001,
    'top_k': 1,
    'max_new_tokens': 128,
    #'max_length': max_tokens,
    #'truncation': True,
    'repetition_penalty': 0.7,
    'prompt_lookup_num_tokens': 10,
    'pad_token_id': tokenizer.eos_token_id
}


'''
generation_params = {
    'do_sample': True,
    'temperature': 0.0000001,
    #'temperature': 0.7,
    'top_p': 0.95,
    'top_k': 40,
    'max_new_tokens': 128,
    #'repetition_penalty': 1.1,
    'repetition_penalty': 0.7,
    'prompt_lookup_num_tokens': 10,
    'pad_token_id': tokenizer.eos_token_id
}
'''

#tokenize_kwargs = {'legacy': False, 'max_length': 2048, 'truncation': True}
#model_kwargs = {'low_cpu_mem_usage': True}

pipe = pipeline(
		'text-generation',
		#device_map='auto',
		model=model,
		tokenizer=tokenizer,
		#model_kwargs=model_kwargs,
		#tokenize_kwargs=tokenize_kwargs,
		**generation_params
		)

def saint_or_sinner_function(entity, keywords, n_res):
	
	### PERFORM SEARCH ###
	
	quotes = ['’‘', '⟫', '“', '』', '「', '”‘', '«', '‘', '«“', ',', '›', '»»', '⟨', '»„', '”', '‚', "'", '„', '」', '⟩', '『', '⟪', '’', '》', '»', '‹', '›‚', '〉', '‹‘', '《', '”„', '"', '〈']
	
	for q in quotes:
		entity = entity.replace(q, '"')
		keywords = keywords.replace(q, '"')
	
	query = entity + ' ' + keywords
	
	t_sleep = 5
	
	#driver.get('https://www.google.com/search?q=' + query.replace(' ', '+') + ' filetype%3AHTML' + '&num=' + str(n_res))
	driver.get('https://www.google.com/search?q=' + query.replace(' ', '+') + ' &tbm=nws&source=lnms&' + 'num=' + str(n_res))
	
	t.sleep(t_sleep)
	# Print search results
	driver.execute_script('document.title="{}";'.format(pdf_name))
	driver.execute_script('window.print();')
	t.sleep(t_sleep)
	# Extract links from the search results
	soup = BeautifulSoup(driver.page_source, 'html.parser')
	#search = soup.find_all('div', class_="yuRUbf")
	search = soup.find_all('div', class_="SoaBEf")
	
	links = []
	for h in search:
		links.append(h.a.get('href'))
		
	### CREATE PDF REPORT ###
	
	w = 150
	h = 4
	l = 1
	
	pdf = FPDF()
	pdf.add_page()
	pdf.add_font('DejaVu', '', '/usr/share/fonts/TTF/DejaVuSansCondensed.ttf', uni=True)
	pdf.add_font('DejaVuB', '', '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf', uni=True)
	pdf.set_font('DejaVuB', '', 8)
	pdf.cell(w, h, txt = 'INTERNET BACKGROUND CHECK REPORT', ln = l, align = 'C')
	pdf.set_font('DejaVu', '', 8)
	pdf.cell(w, h, txt = ' ', ln = l, align = 'C')
	pdf.cell(w, h, txt = 'QUERY: ' + query, ln = l)
	pdf.cell(w, h, txt = 'DATE: ' + today.replace('_', '/') + ' CET ' + hour, ln = l)
	
	# Visit each link from the search results
	
	scores = []
	for i, url in enumerate(links):
		pdf.set_font('DejaVu', '', 8)
		pdf.set_text_color(0, 0, 0)
		# We use Firefox's reader mode to avoid pop-ups
		# and present tidy and identically-formated sites
		driver.get('about:reader?url=' + url)
		t.sleep(t_sleep)
		# Extract the body as plain text
		raw_text = driver.find_element(By.XPATH, "/html/body").text
		
		paragraphs = []
		tokens = 0
		max_tokens = 1536
		for paragraph in raw_text.split('\n'):
			tokens = tokens + len(paragraph)
			if tokens <= max_tokens:
				paragraphs.append(paragraph)
			else:
				break
			
		plain_text = '\n'.join(paragraphs)
		
		pdf.cell(w, h, txt = ' ', ln = l, align = 'L')
		pdf.cell(w, h, txt = 'SOURCE ' + format(i+1, '02'), ln = l)
		if len(plain_text.split('\n'))>1:
			pdf.set_font('DejaVuB', '', 8)
			pdf.set_text_color(0, 0, 0)
			pdf.multi_cell(w, h, txt = plain_text.split('\n')[1])
		if len(plain_text.split('\n'))>0:
			pdf.set_font('DejaVu', '', 8)
			pdf.set_text_color(96, 130, 182)
			pdf.cell(w, h, txt = plain_text.split('\n')[0], ln = l, link = url)
		
		#prompt_template = '<|im_start|>system\n' + system_prompt + '<|im_end|>\n<|im_start|>user\n' + user_prompt + '\n\nCONTEXT:\n' + plain_text + '<|im_end|>\n<|im_start|>assistant\n'
		
		system_prompt = 'You are a helpful, truthful and honest assistant who helps people find information in the provided article. The first word of your response must always be "YES" or "NO".'
		user_prompt = 'Is there any mention of ' + entity + 'in the article below?'
		
		prompt_template=f'''<|im_start|>system
		{system_prompt}<|im_end|>
		<|im_start|>user
		QUESTION: {user_prompt} 
		ARTICLE: {plain_text}
		<|im_end|>
		<|im_start|>assistant
		'''
		
		pipe_output = pipe(prompt_template)[0]['generated_text'].split('<|im_start|>assistant')[1].strip()
		#print(pipe_output[:9])
		
		if not pipe_output.strip().lower().startswith('no'):
			
			### SUMMARY ###
			system_prompt = 'You are a helpful, truthful and honest assistant who helps the use find answers in the provided article.'
			user_prompt = 'Provide a short summary of the article below.'
		
			prompt_template=f'''<|im_start|>system
			{system_prompt}<|im_end|>
			<|im_start|>user
			QUESTION: {user_prompt} 
			ARTICLE: {plain_text}
			<|im_end|>
			<|im_start|>assistant
			'''
			
			pipe_output = pipe(prompt_template)[0]['generated_text'].split('<|im_start|>assistant')[1].strip()
			
			if not pipe_output.endswith('.'):
				pipe_output = '.'.join(pipe_output.split('.')[:-1]) + '.'
				
			#print(pipe_output)
			pdf.set_font('DejaVu', '', 8)
			pdf.set_text_color(0, 0, 0)
			pipe_output = '\n'.join([l.strip() for l in unidecode(pipe_output).splitlines() if l.strip()])
			pdf.multi_cell(w, h, txt = 'ABSTRACT: ' + pipe_output)
			
			### SCORING ###
			system_prompt = f'''
							You are trustworthy risk analyst who evaluates the reputational risk for {entity} on the basis on the provided article.
							Always provide a numeric rating first, followed by a brief justification that should never exceed two sentences.
							Pick up the highest possible rating according to the dictionary below. 
							DICTIONARY = [
							0: "Zero to negligible reputational risk for {entity} or the risk cannot be inferred for {entity}.", 
							1: "Mild reputational risk for {entity} or unsubstantiated claims regarding {entity}.", 
							2: "A trial against {entity} has taken place and the verdict favours {entity}.", 
							3: "A lawsuit against {entity} is mentioned but there is no verdict yet.", 
							4: "{entity} has been found guilty and faces minor penalties.", 
							5: "The reputational risk for {entity} is substantial.",
							6: "The evidence suggests major corruption of {entity} or unlawful behaviour of {entity}."
							7: "{entity} has been found guilty and faces substantial penalties."
							]
							'''
			user_prompt = 'What is the reputational risk that the information in the article below poses for ' + entity + '? Provide the numeric rating first.'
		
			prompt_template=f'''<|im_start|>system
			{system_prompt}<|im_end|>
			<|im_start|>user
			QUESTION: {user_prompt} 
			ARTICLE: {plain_text}
			<|im_end|>
			<|im_start|>assistant
			'''
			
			pipe_output = pipe(prompt_template)[0]['generated_text'].split('<|im_start|>assistant')[1].strip()
			digits = re.findall('\d+', pipe_output)
			try:
				score = int(next(d for d in digits if int(d) < 9))
			except Exception:
				#print(plain_text)
				print(len(plain_text.split('\n')))
				print(len(plain_text.split()))
				print(url)
				score = 3
				pipe_output = 'Not relevant.'
			scores.append(score)
			
			if not pipe_output.endswith('.'):
				pipe_output = '.'.join(pipe_output.split('.')[:-1]) + '.'
				
			pipe_output = '\n'.join([l.strip() for l in unidecode(pipe_output).splitlines() if l.strip()])
			
			pdf.set_font('DejaVuB', '', 8)
			pdf.set_text_color(0, 0, 0)
			pdf.multi_cell(w, h, txt = 'SCORE: ' + str(score))
			pdf.set_font('DejaVu', '', 8)
			pdf.set_text_color(0, 0, 0)
			pdf.multi_cell(w, h, txt = 'RATIONALE: ' + pipe_output)
			
		else:
			score = 0
			scores.append(score)
			pdf.set_font('DejaVu', '', 8)
			pdf.set_text_color(0, 0, 0)
			pdf.multi_cell(w, h, txt = 'SCORE: Not relevant.')
		
		torch.cuda.empty_cache()
	 
	pdf.output('/tmp/report.pdf')
	
	overall_score = round(sum(scores) / n_res, 1)
	#print('OVERALL SCORE:', overall_score)
	
	colours = ['rgb(66, 255, 0)',
	'rgb(177, 207, 0)',
	'rgb(255, 199, 0)',
	'rgb(255, 132, 0)',
	'rgb(255, 112, 0)',
	'rgb(255, 89, 0)',
	'rgb(255, 61, 0)',
	'rgb(255, 0, 0)',
	]
				
	colour = colours[round(sum(scores) / n_res)]
	
	fig = go.Figure(go.Indicator(
	    mode = 'gauge+number',
	    value = overall_score,
	    title = {'text': 'Reputational Risk Score'},
	    domain = {'x': [0, 1], 'y': [0, 1]},
	    number = {'font': {'color': colour}},
	    gauge = {'axis': {'range': [0, 7]}, 'bar': {'color': colour}}
	))
	
	
	pic_path_name = '/tmp/' + query.replace('"', '').replace(' ', '_') + '_' + today + '.png'
	
	fig.write_image(pic_path_name)
	
	### CREATE PDF REPORT ###
	
	w = 150
	h = 4
	l = 1
	
	pdf = FPDF()
	pdf.add_page()
	pdf.add_font('DejaVu', '', '/usr/share/fonts/TTF/DejaVuSansCondensed.ttf', uni=True)
	pdf.add_font('DejaVuB', '', '/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf', uni=True)
	pdf.set_font('DejaVuB', '', 8)
	pdf.cell(w, h, txt = 'INTERNET BACKGROUND CHECK REPORT', ln = l, align = 'C')
	pdf.set_font('DejaVu', '', 8)
	pdf.cell(w, h, txt = ' ', ln = l, align = 'C')
	pdf.cell(w, h, txt = 'QUERY: ' + query, ln = l)
	pdf.cell(w, h, txt = 'DATE: ' + today.replace('_', '/') + ' CET ' + hour, ln = l)
	pdf.cell(w, h, txt = 'SCORE: ' + str(overall_score), ln = l)
	pdf.image(pic_path_name, 10, 40, 150)
	
	
	pdf.output('/tmp/score.pdf')
	
	merger = PdfWriter()
	
	for pdf in ['/tmp/score.pdf', '/tmp/report.pdf', pdf_path]:
	    merger.append(pdf)
	    
	pdf_path_name = '/tmp/' + query.replace('"', '').replace(' ', '_') + '_' + today + '.pdf'
	
	merger.write(pdf_path_name)
	merger.close()
	
	print(entity, overall_score)
	
	return pic_path_name, pdf_path_name
