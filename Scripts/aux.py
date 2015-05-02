import sys,getopt
from os import walk

#DEFAULT VALUES
DEFAULT_DIR = ("../Code/")
DEFAULT_FILE = ""
DEFAULT_SIZE = ( "32" , "64" , "128" , "256" , "512" )
DEFAULT_OUT = "../Results/all.txt"
DEFAULT_EXTENSION = ".cu"

HELP =  "All the programs in the Code directory are execute by:\n\
		python run.py\n\
		To run all files in a specific path use:\n\
		python run.py -p <path> or --path <path>\n\
		To run an exact file use:\n\
		python run.py -f <filename> or --file <filename>\n\
		To run an exact size use:\n\
		python run.py -s <size> --size <size>\
		\n\n\
		All results are output into the Results directory\n\
		To output the results to one file use:\n\
		python run.py -o <filename> --output <filename>"


#GETALL FILES FROM DIRECTORY
def get_files( path , files ) :
	if files != "" :
		return files
	f = []
	for ( dirpath , dirnames , filenames ) in walk( path ) :
		for d in dirnames :
			f.extend( get_files( path + d , files ) )
		f.extend( [ dirpath + "/" + fin[ :-3 ] for fin in filenames \
		if fin[ -3: ] == DEFAULT_EXTENSION ] )
	return f


#PARAMETERS EXTRACTION
def get_params( argv ) :
	directory = DEFAULT_DIR
	files = DEFAULT_FILE
	size = DEFAULT_SIZE
	output = DEFAULT_OUT
	try :
		opts , args = getopt.getopt( argv , "hp:f:s:o:", \
		[ "path=" ,"file=" , "size=", "output=" ] )
	except getopt.GetoptError :
		print HELP
		sys.exit( 2 )
	for opt , arg in opts :
		if opt == "-h" :
			print HELP
			sys.exit()
		elif opt in ( "-p" , "--path" ) :
			directory = arg
		elif opt in ( "-f" , "--file" ) :
			files = arg if arg[ -3: ] != DEFAULT_EXTENSION else arg[ :-3 ]
		elif opt in ( "-s" , "--size" ) :
			size = arg
		elif opt in ( "-o" , "--output" ) :
			output = arg

	return directory , files , size , output

#HACK SEQ FILE
get_size = lambda name, default:("32") if "Seq" in name else default 
