X_c = vec(samples[:,:,1])
Y_c = vec(samples[:,:,2])
Z_c = vec(samples[:,:,3])

scatter(X_c,Z_c)

Y_x = vec(Y[:,:,1])
Y_y = vec(Y[:,:,2])
Y_z = vec(Y[:,:,3])

scatter(Y_x,Y_z)