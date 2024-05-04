import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import pprint
import fast_tsp
from sklearn.cluster import mean_shift
from sklearn.metrics import pairwise_distances
from stqdm import stqdm

import subprocess
import os
import tempfile
from wand.image import Image as wand_image

def main(): 
    st.title("Draw using Complex Fourier Epicycles 🌑🌌")

    if st.button("Under the hood"):

        st.header("The discrete fourier transform")

        st.latex(r"""\begin{aligned}
        X_{k}&={\dfrac{1}{N}\sum _{n=0}^{N-1}x_{n}\cdot e^{-i k {\frac {2\pi }{N}}n}}\\
        &={ \dfrac{1}{N}\sum _{n=0}^{N-1}x_{n}\left[\cos \left(k{\frac {2\pi }{N}}n\right)-i\, \sin \left({k\frac {2\pi
        }{N}}n\right)\right],}
        \end{aligned}""")

        st.text("""
        Xk is the amount of kth frequency in the signal, a complex number 
            (with amplitude and phase)
        N is the number of time samples
        n is the current sample we are on, i.e. 0 to N-1
        x_n is the value of the signal at time n, a fourier coefficient
        t is the time, which ranges from 0 to 2pi
                
        If we connect all epicycles from tip-to-tip at their initial conditions,
                as all epicycles of various amplitudes rotate at their frequencies,
                the last epicycle effectively "draws" the image
        """)

        st.caption("""
        Useful Links:
        1. https://www.dynamicmath.xyz/fourier-epicycles/\n
        2. https://www.myfourierepicycles.com\n
        3. Mathologer: https://www.google.com/url?sa=t&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwittIy3seGDAxU8mmoFHcLpDmIQtwJ6BAgdEAI&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DqS4H6PEcCCA&usg=AOvVaw3d9k_lTkWUlrvj2ETlb-ww&opi=89978449\n
        4. GoldPlatedGoof: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwittIy3seGDAxU8mmoFHcLpDmIQtwJ6BAgaEAI&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D2hfoX51f6sg&usg=AOvVaw27HmdxTn2HNtMgjBOH-IYs&opi=89978449\n
        5. The Coding Train: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwittIy3seGDAxU8mmoFHcLpDmIQtwJ6BAgeEAI&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D0b3R8oWffkw&usg=AOvVaw0zSjV3TzxiVuZZpsjjZkgb&opi=89978449\n
        6. 3B1B: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwizv8zUseGDAxVWkyYFHRvEA2UQtwJ6BAgbEAI&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Dr6sGWTCMz2k&usg=AOvVaw2pSPj4IUnE9g8P4eUsqB9Q&opi=89978449\n
        7. https://en.wikipedia.org/wiki/Fourier_series
        """)

        st.subheader("Main App")

    st.info("First, convert your image to SVG using: https://picsvg.com")

    st.caption("*Why? A 2D b&w line art style is required for drawing*")

    st.info("Then, upload your SVG:")

    svg_file = st.file_uploader("Upload:", type=["svg"])

    if svg_file is not None:
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(svg_file.read())
            temp_file.seek(0)

        print(temp_file, temp_file.name)
        """

        with wand_image( filename = svg_file.name ) as png:
            png.format = 'png'
            png.save(filename='out.png')
        
        img = Image.open('out.png')

        st.image(img, caption='Uploaded')

        # preprocessing

        # resize
        if img.size[0] > 300 and img.size[1] > 300:
            img = img.resize((300, 300))

        # find contours
        img = img.filter(ImageFilter.CONTOUR)

        st.image(img, caption = "Processed")

        st.info("Now, imagine you were about to hand-draw the image. To properly draw it, how much time would you need?")

        complexity = st.radio(label = "Difficulty:", options=['Low', "Medium", 'High'], index = 1 )
        st.caption("*If unsure, leave as Medium*")
        
        if st.button("Let's draw using math!"):

            with st.spinner("Computing..."):
                
                # extract path coordinates
                xy_coords = np.flip(np.column_stack(np.where(np.array(img) < 10)), axis = 0)
                # print(xy_coords, len(xy_coords))

                # Mean Shift
                if complexity == 'Low':
                    start = 3
                    stop = 5
                    step = 0.25
                    points = 500

                if complexity == 'Medium':
                    start = 2
                    stop = 4
                    step = 0.25
                    points = 1300

                elif complexity == 'High':
                    start = 1
                    stop = 3
                    step = 0.25
                    points = 2300

                for b in np.arange(start, stop, step):
                    xy_coords, labels = mean_shift(xy_coords, bandwidth = b)
                    # st.write('mean shift,', str(b) + ": " + str(len(xy_coords)))
                    if len(xy_coords) <= points:
                        st.success("Mean shift completed!")
                        break
            
                # TSP 
                dists = pairwise_distances(xy_coords).astype(np.int64)
                tsp_idxs = fast_tsp.find_tour(dists)
                tsp_xy_coords = []
                for i in tsp_idxs:
                    tsp_xy_coords.append(xy_coords[i])
                st.success("Traveling salesman problem solved!")

                # DFT 
                z = [] 
                for i in range(0, len(tsp_xy_coords)):
                    z.append(complex(tsp_xy_coords[i][0], tsp_xy_coords[i][1]))

                def dft(x):
                    N = len(x)
                    X = []
                    for k in range(N):
                        z = 0 + 0j
                        for n in range(N):
                            phi = -2j*np.pi*k*n/N
                            z += x[n] * np.exp(phi)
                        freq = k 
                        z = -1j * z 
                        amp = (np.sqrt(np.real(z)**2 + np.imag(z)**2)/N)
                        phase = np.arctan2(np.imag(z), np.real(z)) # rad
                        X += [{'z': z, 'freq':freq, 'amp':amp, 'phase':phase}]
                    return eval(pprint.pformat(sorted(X, key = lambda z: z['amp'], reverse=True))) # sorted by amp
                fourier = dft(z)
                st.success("Discrete fourier coefficients computed!")

            # animate
            st.header("Epicycle Animation")

            st.info("Finally, wait for your animation. This will take some time, but it will be worth it! ⭐")

            with st.spinner("Creating animation..."):
  
                # blank window
                fig = plt.figure()
                fig.set_dpi(100)
                fig.set_size_inches(8, 8)
                ax = plt.axes(xlim = (-300, 300), ylim = (-300, 300))
                ax.set_xticks([])
                ax.set_yticks([])
                plt.suptitle("https://complex-fourier-draw.streamlit.app")

                # epicycles 
                # (add arrows next)

                # initialize
                patches = []
                for i in range(len(fourier)):
                    patches.append(plt.Circle((0, 0), fourier[i]['amp'], fill = False))

                # final drawing line
                line, = ax.plot([], [], lw = 2)
                patches.append(line) # important

                def init():
                    for p in patches[:-1]:
                        ax.add_artist(p)
                    line.set_data([], [])
                    return patches

                # init empty values for x and y coordinates for final drawing line
                xdata, ydata = [], []

                init()

                # animation

                # with progress bar

                # corrected epicycle alignment

                for i in stqdm(range(len(fourier))):
                
                    t = i * (2 * np.pi / len(fourier))
                    
                    for idx in range(len(fourier)-1):        

                        if idx == 0:
                            
                            radius = fourier[0]['amp']

                            freq = fourier[0]['freq']

                            phase = fourier[0]['phase']

                            x = radius * np.cos(t * freq + phase)

                            y = radius * np.sin(t * freq + phase)

                            patches[0].center = (x, y)

                        else:

                            prev_x, prev_y = patches[idx].center

                            radius = fourier[idx]['amp']

                            freq = fourier[idx]['freq']

                            phase = fourier[idx]['phase']

                            x = prev_x + radius * np.cos(t * freq + phase)

                            y = prev_y + radius * np.sin(t * freq + phase)

                            patches[idx + 1].center = (x, y)

                    # add values to x and y holders of final patch for drawing
                        
                    xdata.append( x )

                    ydata.append( y )

                    line.set_data(xdata, ydata)
                    
                    fig.savefig(str(i) + '.png')

            with st.spinner("Compiling animation..."):

                images = []
                for i in range(len(fourier)):
                    
                    exec('a'+str(i)+'=Image.open("'+str(i)+'.png")')
                    images.append(eval('a'+str(i)))
                
                images[0].save('output.gif',
                            save_all = True,
                            append_images = images[1:],
                            duration = 120,
                            loop = 1)

                st.balloons()
                st.success("Animation ready! 😊")

                st.caption("(right click to download the gif)")

            # removing temp files
            for i in range(len(fourier)):
                
                os.remove(str(i)+'.png')
            
            # show gif
            st.image('output.gif')

            st.caption("If gif is too slow, speed up here: https://onlinegiftools.com/make-gif-faster")

    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("By: Arifeen Saeed")

if __name__ == '__main__':
    main()
