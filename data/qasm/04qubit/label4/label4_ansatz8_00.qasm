OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.9996262021528226) q[0];
ry(2.871654415712239) q[1];
cx q[0],q[1];
ry(-2.77972606516157) q[0];
ry(-2.397281134652263) q[1];
cx q[0],q[1];
ry(1.4502407748224393) q[2];
ry(-2.9597433822305743) q[3];
cx q[2],q[3];
ry(1.983518592046291) q[2];
ry(1.381480835845629) q[3];
cx q[2],q[3];
ry(-0.8206723873345521) q[0];
ry(-0.054334361899141925) q[2];
cx q[0],q[2];
ry(-1.0965273409547651) q[0];
ry(0.6062847034860637) q[2];
cx q[0],q[2];
ry(1.8847792051189953) q[1];
ry(-1.8817008578817616) q[3];
cx q[1],q[3];
ry(-2.2059486517597935) q[1];
ry(2.0961681761620685) q[3];
cx q[1],q[3];
ry(2.2723646563676976) q[0];
ry(1.7529531734125756) q[1];
cx q[0],q[1];
ry(-3.1176194839120384) q[0];
ry(1.1165010056082811) q[1];
cx q[0],q[1];
ry(1.2897868178488774) q[2];
ry(2.508596161713501) q[3];
cx q[2],q[3];
ry(0.6418272218869053) q[2];
ry(1.8372132579746652) q[3];
cx q[2],q[3];
ry(-3.100292876191166) q[0];
ry(0.5440108435529164) q[2];
cx q[0],q[2];
ry(0.7070121674564287) q[0];
ry(1.698934865373108) q[2];
cx q[0],q[2];
ry(2.4820729342130754) q[1];
ry(2.7711183073967147) q[3];
cx q[1],q[3];
ry(0.09165273323656642) q[1];
ry(0.7627043188034913) q[3];
cx q[1],q[3];
ry(2.5436947423984138) q[0];
ry(0.7770937498518453) q[1];
cx q[0],q[1];
ry(-2.4377443642328385) q[0];
ry(2.365559830658923) q[1];
cx q[0],q[1];
ry(-0.028833575830160108) q[2];
ry(0.6309550459635102) q[3];
cx q[2],q[3];
ry(0.2269356559522162) q[2];
ry(-3.017341540278916) q[3];
cx q[2],q[3];
ry(2.7595692684655853) q[0];
ry(-0.9682980986471152) q[2];
cx q[0],q[2];
ry(-2.5299698859308726) q[0];
ry(-0.1378370719767208) q[2];
cx q[0],q[2];
ry(-0.1367527349613561) q[1];
ry(-1.569919075440283) q[3];
cx q[1],q[3];
ry(-0.601894570408593) q[1];
ry(-1.5151671839967191) q[3];
cx q[1],q[3];
ry(0.324088677416178) q[0];
ry(-1.7791805926530655) q[1];
ry(-2.8129150900586226) q[2];
ry(0.7984574308090258) q[3];