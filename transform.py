import numpy as np
import pandas as pd
import sys
import os
import re


def parse_args(argv):
	if len(sys.argv) > 2:
		raw_path = sys.argv[1]
		dest_file_path = sys.argv[2]
		isPostive = None
		if len(sys.argv) > 3:
			if sys.argv[3] == 'P':
				isPostive = True
			elif sys.argv[3] == 'N':
				isPostive = False
			else:
				isPostive = None
		return (raw_path, dest_file_path, isPostive)
	else:
		print "Please input raw data path and transformed data file path."
        exit(1)

def extract_features(device_set, raw_data_B_map, raw_data_F_map, isPostive):
	all_features = list()

	feature_titles = range(32 + 1)
	unprocess_size = len(device_set)
	process_counter = 0
	print "progress:%.2f%%, %d in %d" % (process_counter * 100.0 / unprocess_size, process_counter, unprocess_size)

	for device in device_set:
		if raw_data_B_map.has_key(device) and raw_data_F_map.has_key(device):
			raw_abspath_B = raw_data_B_map[device]
			raw_abspath_F = raw_data_F_map[device]
			df_B = pd.read_csv(raw_abspath_B, header = 0)
			df_F = pd.read_csv(raw_abspath_F, header = 0)
			features = [0] * (32 + 1)

			df_B_0 = df_B[df_B.columns[0]]
			df_B_1 = df_B[df_B.columns[1]]
			features[0] = df_B_0.mean()
			features[1] = df_B_0.std()
			features[2] = df_B_0.median()
			features[3] = df_B_0.mad()
			features[4] = df_B_0.skew()
			features[5] = df_B_0.kurtosis()
			features[6] = df_B_0.max()
			features[7] = df_B_0.min()

			features[8] = df_B_1.mean()
			features[9] = df_B_1.std()
			features[10] = df_B_1.median()
			features[11] = df_B_1.mad()
			features[12] = df_B_1.skew()
			features[13] = df_B_1.kurtosis()
			features[14] = df_B_1.max()
			features[15] = df_B_1.min()

			df_F_0 = df_F[df_F.columns[0]]
			df_F_1 = df_F[df_F.columns[1]]

			features[16] = df_F_0.mean()
			features[17] = df_F_0.std()
			features[18] = df_F_0.median()
			features[19] = df_F_0.mad()
			features[20] = df_F_0.skew()
			features[21] = df_F_0.kurtosis()
			features[22] = df_F_0.max()
			features[23] = df_F_0.min()

			features[24] = df_F_1.mean()
			features[25] = df_F_1.std()
			features[26] = df_F_1.median()
			features[27] = df_F_1.mad()
			features[28] = df_F_1.skew()
			features[29] = df_F_1.kurtosis()
			features[30] = df_F_1.max()
			features[31] = df_F_1.min()

			features[32] = device
			all_features.append(features)
			process_counter = process_counter + 1
			print "progress device %s:%.2f%%, %d in %d" % (device, process_counter * 100.0 / unprocess_size, process_counter, unprocess_size)
		else:
			print "Please check raw data file about device %s, it should be include 2 files(B and F)." % (device)
			exit(1)
	return (feature_titles, all_features)

def main(raw_path, dest_file_path, isPostive):
	raw_abspaths = list()

	if isPostive is None:
		target = None
	else:
		if isPostive:
			target = 1
		else:
			target = 0

	if os.path.exists(raw_path):
		if os.path.isfile(raw_path):
			raw_abspaths.append(os.path.abspath(raw_path))
		if os.path.isdir(raw_path):
			for sub_raw_path in os.listdir(raw_path):
				raw_abspaths.append(os.path.abspath(raw_path + '/' + sub_raw_path))

	raw_data_B_map = dict()
	raw_data_F_map = dict()
	device_set = set()

	file_regx = '([0-9a-zA-Z\-]+)_([BF]{1})\.csv'
	for raw_abspath in raw_abspaths:
		filename = os.path.basename(raw_abspath)
		file_detail = re.findall(file_regx, filename)
		direct = file_detail[0][1]
		device = file_detail[0][0]

		print 'direct:%s, device:%s' % (direct, device)
		device_set.add(device)
		if direct == 'B':
			raw_data_B_map[device] = raw_abspath
		if direct == 'F':
			raw_data_F_map[device] = raw_abspath

	print "device total: %d" % (len(device_set))
	features = extract_features(device_set, raw_data_B_map, raw_data_F_map, isPostive)
	feature_titles = features[0]
	feature_data = features[1]

	if target is None:
		pass
	else:
		feature_titles.append('qualified')

	feature_np_array = np.array(feature_data)
	if target is None:
		feature_target_np_array = feature_np_array
	else:
		target_np_arrray = np.array([target for i in range(len(feature_data))], dtype = 'int64')
		feature_target_np_array = np.column_stack([feature_np_array, target_np_arrray])

	feature_df = pd.DataFrame(data = feature_target_np_array, columns = feature_titles)
	feature_df.to_csv(dest_file_path, index = False)


if __name__ == '__main__':
    argvs = parse_args(sys.argv)
    main(raw_path = argvs[0], dest_file_path = argvs[1], isPostive = argvs[2])