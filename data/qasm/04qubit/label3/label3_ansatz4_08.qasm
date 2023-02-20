OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.4457093833493664) q[0];
rz(1.5044153510009044) q[0];
ry(0.551532840410637) q[1];
rz(0.3942241630300597) q[1];
ry(0.8624027664650821) q[2];
rz(0.8396732179224412) q[2];
ry(-1.2647171690519416) q[3];
rz(-0.8121990903896483) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.9166796409344007) q[0];
rz(-1.3866772313031246) q[0];
ry(1.469806035161761) q[1];
rz(-0.006028533868391325) q[1];
ry(-2.2483386387268727) q[2];
rz(-1.12003403254787) q[2];
ry(0.5638476663449117) q[3];
rz(1.1539177891176795) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.8331398780783488) q[0];
rz(-1.799278878786779) q[0];
ry(2.867775126161359) q[1];
rz(2.9087783691069142) q[1];
ry(-2.4740768037098655) q[2];
rz(-0.33276724836496496) q[2];
ry(0.999680733352152) q[3];
rz(0.5787944780346725) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.2851051944649505) q[0];
rz(-2.8788481563282544) q[0];
ry(-1.8633928762780148) q[1];
rz(-0.41869017160078886) q[1];
ry(2.826226214907398) q[2];
rz(-0.12058166563500361) q[2];
ry(0.18309675156343264) q[3];
rz(-3.1262219249513574) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.4853466936952135) q[0];
rz(-2.736654685113762) q[0];
ry(1.2875050680941886) q[1];
rz(-2.919146850005487) q[1];
ry(-0.9730764827885326) q[2];
rz(0.05689651203164826) q[2];
ry(1.8066319101810429) q[3];
rz(-3.0143350870826877) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.1492972385380225) q[0];
rz(0.9594691124665794) q[0];
ry(-0.949476601825912) q[1];
rz(1.648553425370145) q[1];
ry(0.5513403478320313) q[2];
rz(-0.5662833384237941) q[2];
ry(0.5291906657924774) q[3];
rz(0.2026249091746015) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.6940019855175885) q[0];
rz(-1.2132705187286463) q[0];
ry(2.366792700959642) q[1];
rz(2.7070674722700168) q[1];
ry(-0.8763725636277601) q[2];
rz(2.9413960813706805) q[2];
ry(1.6533076232181572) q[3];
rz(0.9750694146048259) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.451119171950322) q[0];
rz(2.0284815913680756) q[0];
ry(-1.535368445910947) q[1];
rz(-1.0692126122803018) q[1];
ry(-0.8091536423023591) q[2];
rz(-0.6696237408107093) q[2];
ry(-2.07615908375564) q[3];
rz(2.901490285479039) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.396596756229509) q[0];
rz(0.9495119495723962) q[0];
ry(-1.9162846172288122) q[1];
rz(2.5422816475411736) q[1];
ry(2.278153959550849) q[2];
rz(-3.1217042503865846) q[2];
ry(0.7421008357713746) q[3];
rz(-1.1501877703786674) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.2424583641372955) q[0];
rz(-2.1146355305358844) q[0];
ry(-1.2344546283354179) q[1];
rz(2.945739304388321) q[1];
ry(1.172375703633613) q[2];
rz(-2.4774205702746372) q[2];
ry(-0.3354103628827368) q[3];
rz(2.489725854451527) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.5110450556987585) q[0];
rz(-1.4015136112256388) q[0];
ry(0.8079991867651367) q[1];
rz(-0.19624336834398992) q[1];
ry(2.8986304877143234) q[2];
rz(2.350030915794728) q[2];
ry(1.4275643519170993) q[3];
rz(-0.46163920574061107) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.9130482500829995) q[0];
rz(0.24532054019371682) q[0];
ry(-2.242719595725034) q[1];
rz(-3.1168688362559487) q[1];
ry(-1.7754560725321011) q[2];
rz(2.22414060837772) q[2];
ry(2.5564787003530633) q[3];
rz(0.5086353128549701) q[3];