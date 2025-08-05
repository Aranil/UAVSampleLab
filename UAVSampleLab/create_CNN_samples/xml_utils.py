'''
Script to generate metafile for sample patch mosaic in xml format
#https://lxml.de/tutorial.html

'''


# import module of different version
try:
  from lxml import etree
  print("running with lxml.etree")
except ImportError:
  try:
    # Python 2.5
    print("running with cElementTree on Python 2.5+")
  except ImportError:
    try:
      # Python 2.5
      print("running with ElementTree on Python 2.5+")
    except ImportError:
      try:
        # normal cElementTree install
        import cElementTree as etree
        print("running with cElementTree")
      except ImportError:
        try:
          # normal ElementTree install
          import elementtree.ElementTree as etree
          print("running with ElementTree")
        except ImportError:
          print("Failed to import ElementTree from any known place")



filetype = "raster"



def generate_xml_meta(
                    raster,
                    chanell_meta,
                    meta_dict,
                    ofile
                    ):

    '''
    raster_meta: dict

    chanell_meta: dict
        information to channel names
        chanell_dict = CIR16 = {   1: 'A',
                                   2: 'B',
                                    ...
                                   15: 'V',
                                   16: 'Y'
                               }

    ofile: str
        output file name
    '''


    # part 1 -start with  header of the metafile
    root = etree.XML('<sample-set></sample-set>')

    file_h1 = etree.SubElement(root, "file-information")

    etree.SubElement(file_h1, "{}-file".format(filetype), name="{}".format(raster['fpath']))
    etree.SubElement(file_h1, "{}-info".format(filetype),
                                     size="{}".format(raster['size']),
                                     channels="{}".format(raster['bnumber']),
                                     data_type=raster['dtype']
                                 )


    #for child in root[0]:
    #    print(child.tag)

    # add subelements -> channel names
    file_h2 = etree.SubElement(root, "channels")
    for k, v in chanell_meta.items():
        etree.SubElement(file_h2, "name_{}".format(k)).text = v

    #tree = etree.ElementTree(root)

    # get access to tags
    #print(root.tag)

    # get access to child
    #child = root[0]
    #print(child.tag)

    etree.indent(root, space="    ")

    # add part 2 - samples
    #root.insert(1, etree.Element("sample-space"))
    #etree.SubElement(root, "sample-info type", name="raw").text = "WR_5_0_1"
    class_m = etree.SubElement(root, "class-mapping")


    start = root[:1]
    end = root[-1:]
    #print(start[0].tag)
    #print(end[0].tag)


    #crop_type_code = "WR"
    #pattern_code = "WR_5_0_1"
    #pattern = "lodged_area"
    #fid = '190-00'
    #file_link = 'D:/__crop_samples/stacks/samples_tiles/WR/WR_6_0_2/190_WR_6_0_2_79.tif'

    #segment_meta['flink']
    #segment_meta['pattern_code']
    #segment_meta['pattern']
    #segment_meta['fid']

    for column_position, segment_meta in meta_dict.items():

        if isinstance(segment_meta, dict):
            #etree.SubElement(class_m, "crop_type_code").text = "{}".format(crop_type_code)
            etree.SubElement(class_m, "pattern_code",
                                                     column_position=str(column_position),
                                                     name=segment_meta['pattern'],
                                                     field_amount=str(segment_meta['fid_amount']),
                                                     samples_number= str(len(segment_meta['flink'])),
                                                     crop=segment_meta['crop_type_code']
                                                     ).text = str(segment_meta['pattern_subcode'])

            #etree.SubElement(class_m, "pattern").text = pattern
            class_fid = etree.SubElement(class_m, "fields")

            for fid, fpath_list in segment_meta['fid_path_collection'].items():

                etree.SubElement(class_fid, "fields",
                                 #name=segment_meta['fid_list'][i],
                                 sl_nr=fid,
                                 segment_amount=str(len(fpath_list))
                                 #samples_amount=str(segment_meta['sample_number']),
                                 #segment_method='ChessboardSegment30'
                                 )

                class_sample_numb = etree.SubElement(class_fid, "segments")


                for k, v in enumerate(fpath_list):
                    etree.SubElement(class_sample_numb, "segment_{}".format(k)).text = v

                tree = etree.ElementTree(root)
                etree.indent(root, space="    ")

        elif isinstance(segment_meta, str):
            etree.SubElement(class_m, "pattern_code",
                             column_position=str(column_position),
                             name=segment_meta
                             )
            tree = etree.ElementTree(root)
            etree.indent(root, space="    ")

        etree.tostring(root, xml_declaration=True)
        etree.tostring(root, encoding='iso-8859-1')


        # part 3 - write content to file
        tree.write(ofile)

