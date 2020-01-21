step = 2
if self.jump_level < step:
    while satisfied_with_coreg is False:
        coregister_ = input("Step 2 - Co-Register with Euclidean Transform: Proceed? (y/n): ")
        if coregister_.lower() == "y":
            if self.warp_matrix is None:
                self.current_frame_a, self.current_frame_b = self.grab_frames()
            else:
                self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=self.warp_matrix)
            self.frame_count += 1
            a_as_16bit = bdc.to_16_bit(self.current_frame_a)
            b_as_16bit = bdc.to_16_bit(self.current_frame_b)
            cv2.imshow("A", a_as_16bit)
            cv2.imshow("B Prime", b_as_16bit)
            continue_stream = self.keep_streaming()
            continue_stream = True
            a_8bit = bdc.to_8_bit(self.current_frame_a)
            b_8bit = bdc.to_8_bit(self.current_frame_b)
            warp_ = ic.get_euclidean_transform_matrix(a_8bit, b_8bit)
            self.warp_matrix = warp_

            a, b, tx = warp_[0][0], warp_[0][1], warp_[0][2]
            c, d, ty = warp_[1][0], warp_[1][1], warp_[1][2]

            print("\tTranslation X:{}".format(tx))
            print("\tTranslation Y:{}\n".format(ty))

            scale_x = np.sign(a) * (np.sqrt(a ** 2 + b ** 2))
            scale_y = np.sign(d) * (np.sqrt(c ** 2 + d ** 2))

            print("\tScale X:{}".format(scale_x))
            print("\tScale Y:{}\n".format(scale_y))

            phi = np.arctan2(-1.0 * b, a)
            print("\tPhi Y (rad):{}".format(phi))
            print("\tPhi Y (deg):{}\n".format(np.degrees(phi)))

            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=self.warp_matrix)
            temp_a_8bit = np.array(self.current_frame_a, dtype='uint8')  # bdc.to_8_bit()
            temp_b_prime_8bit = np.array(self.current_frame_b, dtype='uint8')
            # temp_b_prime_8bit = bdc.to_8_bit(self.current_frame_b)
            GOOD_MATCH_PERCENT = 0.10
            orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20)
            keypoints1, descriptors1 = orb.detectAndCompute(temp_a_8bit, None)
            keypoints2, descriptors2 = orb.detectAndCompute(temp_b_prime_8bit, None)

            print("A has {} key points".format(len(keypoints1)))
            print("B has {} key points".format(len(keypoints2)))
            # cv2.drawMatchesKnn expects list of lists as matches.

            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(descriptors1, descriptors2, None)
            matches.sort(key=lambda x: x.distance, reverse=False)

            # BFMatcher with default params
            bf = cv2.BFMatcher()
            knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            lowe_ratio = 0.89

            # Apply ratio test
            good_knn = []

            for m, n in knn_matches:
                if m.distance < lowe_ratio * n.distance:
                    good_knn.append([m])

            print("Percentage of Matches within Lowe Ratio of 0.89: {0:.4f}".format(
                100 * float(len(good_knn)) / float(len(knn_matches))))

            imMatches = cv2.drawMatches(temp_a_8bit, keypoints1, temp_b_prime_8bit, keypoints2, matches[:25], None)
            cv2.imshow("DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING",
                       cv2.resize(imMatches, (int(imMatches.shape[1] * 0.5), int(imMatches.shape[0] * 0.5))))
            cv2.waitKey(60000)
            cv2.destroyAllWindows()

            y_or_n = input("Are you satisfied with coregistration? (y/n)")
            if y_or_n.lower() == 'y':
                satisfied_with_coreg = True
                continue_stream = True
                self.warp_matrix = None
                self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=self.warp_matrix)

            if y_or_n.lower() == 'n':
                continue_stream = False
                satisfied_with_coreg = False
                self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=self.warp_matrix)

            cv2.destroyAllWindows()

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=self.warp_matrix)
            a_as_16bit = bdc.to_16_bit(self.current_frame_a)
            b_as_16bit = bdc.to_16_bit(self.current_frame_b)
            cv2.imshow("A", a_as_16bit)
            cv2.imshow("B Prime", b_as_16bit)
            continue_stream = self.keep_streaming()

cv2.destroyAllWindows()


