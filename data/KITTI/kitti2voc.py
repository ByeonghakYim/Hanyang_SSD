import xml.etree.ElementTree as ET
import os
import argparse
import cv2
import shutil

if __name__ =="__main__":
	print "Create xml type annotation files"

	parser = argparse.ArgumentParser(description="Create AnnotatedDatum database")
	parser.add_argument("--data-root-dir", help="KITTI data root directory")
	parser.add_argument("--duplicate-car", help="The number of duplicatoin of car")
	parser.add_argument("--duplicate-ped", help="The number of duplicatoin of pedestrian")
	parser.add_argument("--duplicate-cyc", help="The number of duplicatoin of cyclist")

	args = parser.parse_args()

	data_root_dir = args.data_root_dir
	iter_car = args.duplicate_car
	iter_ped = args.duplicate_ped
	iter_cyc = args.duplicate_cyc

	img_dir = data_root_dir + '/img/'
	anno_dir = data_root_dir + '/anno/'
	out_dir = data_root_dir + '/Annotations/'

	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)

	os.mkdir(out_dir, 0755)

	list_file = data_root_dir + '/trainval.txt'

	f = open(list_file)

	while True:
		line = f.readline()
		if not line: break
		output = out_dir + line.rstrip() + '.xml'
		img_name = img_dir + line.rstrip() + '.png'
		img = cv2.imread(img_name)
		height, width, depth = img.shape

		annotation = ET.Element("annotation")
		folder = ET.SubElement(annotation, "folder").text = "KITTI"
		filename = ET.SubElement(annotation, "filename").text = line.rstrip() + '.png'
		size = ET.SubElement(annotation, "size")
		segmented = ET.SubElement(annotation, "segmented").text = "0"

		ET.SubElement(size, "width").text = str(width)
		ET.SubElement(size, "height").text = str(height)
		ET.SubElement(size, "depth").text = "3"


		anno = open(anno_dir + line.rstrip() + '.txt')

		while True:
			gt_line = anno.readline()
			if not gt_line: break
			obj_gt = gt_line.split(" ")
			if (obj_gt[0] == "Pedestrian") or (obj_gt[0] == "Person_sitting"):
				num_iter = int(iter_ped)
			elif (obj_gt[0] == "Car") or (obj_gt[0] == "Van") or (obj_gt[0] == "Truck"):
				num_iter = int(iter_car)
			elif (obj_gt[0] == "Cyclist"):
				num_iter = int(iter_cyc)
			else:
				num_iter = 1

			for i in xrange(num_iter):
				obj = ET.SubElement(annotation, "object")

				obj_name = ET.SubElement(obj, "name").text = obj_gt[0]
				obj_truncated = ET.SubElement(obj, "truncated").text = obj_gt[1]
				obj_occluded = ET.SubElement(obj, "occluded").text = obj_gt[2]
				obj_alpha = ET.SubElement(obj, "alpha").text = obj_gt[3]
				obj_bndbox = ET.SubElement(obj, "bndbox")
				obj_dimensions = ET.SubElement(obj, "dimensions")
				obj_location = ET.SubElement(obj, "location")
				obj_rotation_y = ET.SubElement(obj, "rotation_y").text = "0"
				obj_bndbox_xmin = ET.SubElement(obj_bndbox, "xmin").text = str(int(float(obj_gt[4])))
				obj_bndbox_ymin = ET.SubElement(obj_bndbox, "ymin").text = str(int(float(obj_gt[5])))
				obj_bndbox_xmax = ET.SubElement(obj_bndbox, "xmax").text = str(int(float(obj_gt[6])))
				obj_bndbox_ymax = ET.SubElement(obj_bndbox, "ymax").text = str(int(float(obj_gt[7])))
				obj_dimensions_h = ET.SubElement(obj_dimensions, "height").text = obj_gt[8]
				obj_dimensions_w = ET.SubElement(obj_dimensions, "width").text = obj_gt[9]
				obj_dimensions_l = ET.SubElement(obj_dimensions, "length").text = obj_gt[10]
				obj_location_x = ET.SubElement(obj_location, "x").text = obj_gt[11]
				obj_location_y = ET.SubElement(obj_location, "y").text = obj_gt[12]
				obj_location_z = ET.SubElement(obj_location, "z").text = obj_gt[13]

		tree = ET.ElementTree(annotation)
		tree.write(output)
		anno.close()
	f.close()