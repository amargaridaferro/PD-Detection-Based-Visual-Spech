

def get_sec(time_str):
    """Get seconds from time."""
    m, s = time_str.split(':')
    return int(m) * 60 + int(s)

def organize_videos_details(header,ts):
    list = []
    ts_list = []
    with open(header, 'r') as f:
        i = 1
        for row in f:
            # row variable is a list that represents a row in csv
            row = row.strip('\n')
            if(i<139):
                if i < 10:
                    row += ",p_00" + str(i) + ','  # , to process in the sh file everything well
                    i += 1
                    list.append(row)
                elif(i<100 and i>=10):
                    row += ",p_0" + str(i) +','  # , to process in the sh file everything well
                    i += 1
                    list.append(row)
                else:
                    row += ",p_" + str(i) + ','  # , to process in the sh file everything well
                    i += 1
                    list.append(row)
        f.close()

        with open(ts, 'r') as g:
            for row in g:
                #row = row.strip('\n')
                #row = row.replace("-",",")
                #row += ','

                #Get the duration of each video
                #timestamp = row.split(",")
                #ti = get_sec(timestamp[0])
                #tf = get_sec(timestamp[1])
                #totalsec = tf-ti
                #row+=str(totalsec)

                ts_list.append(row)
        g.close()

        for index in range(len(list)):
            line1 = list[index].strip('\n')
            line2 = ts_list[index].strip('\n')
            line = line1 + line2
            print(line)


if __name__ == '__main__':
    organize_videos_details('../videos_PD_test.csv','../times_stamps.csv')
