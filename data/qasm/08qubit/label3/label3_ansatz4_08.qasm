OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(3.141358562287101) q[0];
rz(-0.958221521318438) q[0];
ry(-1.571100562163308) q[1];
rz(-1.701603494545342) q[1];
ry(-2.773792777332671) q[2];
rz(1.5696561496566115) q[2];
ry(1.5861275674517052) q[3];
rz(-1.5700374371370585) q[3];
ry(-1.6338100497340324) q[4];
rz(1.570435023865625) q[4];
ry(1.5039812540723974) q[5];
rz(-1.5704967144975726) q[5];
ry(-3.1415170412365945) q[6];
rz(-2.7158144569709615) q[6];
ry(1.5634794466611346) q[7];
rz(-0.0002731112274835029) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.4464914473873627) q[0];
rz(-3.1414626077756584) q[0];
ry(-1.5719838226680327) q[1];
rz(3.1400044271668044) q[1];
ry(-1.5703855352649905) q[2];
rz(-2.64134367190076) q[2];
ry(-1.5402519298635144) q[3];
rz(1.585380192063705) q[3];
ry(-1.570469290999196) q[4];
rz(0.7498839100667913) q[4];
ry(-1.5705269573062575) q[5];
rz(-0.840141045789755) q[5];
ry(2.472407153999626) q[6];
rz(-3.1209121231334502) q[6];
ry(1.5726837095898027) q[7];
rz(-0.005026474481088172) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5686526584216287) q[0];
rz(3.1402697015902965) q[0];
ry(1.5704883342825253) q[1];
rz(-0.2541108987894597) q[1];
ry(-0.001490288480225226) q[2];
rz(-2.0786603534946337) q[2];
ry(-1.567635159680255) q[3];
rz(0.24667334694902368) q[3];
ry(-1.3879124004611958) q[4];
rz(-1.2602520481646442) q[4];
ry(-1.5721636038671054) q[5];
rz(1.5871265810533428) q[5];
ry(-3.0984205232212423) q[6];
rz(-0.4768361710092037) q[6];
ry(1.5704492140556123) q[7];
rz(-1.570470659113989) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.549382561031086) q[0];
rz(-3.1413775480730552) q[0];
ry(-3.131242044516699) q[1];
rz(-0.5270721921693017) q[1];
ry(-1.5710039687683848) q[2];
rz(-1.8024503279528972) q[2];
ry(-1.5621537324436658) q[3];
rz(3.136524814843699) q[3];
ry(-1.5671343416939854) q[4];
rz(-3.094349551619823) q[4];
ry(-1.5812129264081232) q[5];
rz(-2.8385505957354744) q[5];
ry(0.00018649846959117866) q[6];
rz(2.058197971020689) q[6];
ry(1.581016767956422) q[7];
rz(-1.5633338252052797) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5846971598921495) q[0];
rz(-3.088110485897341) q[0];
ry(-1.5724724255499207) q[1];
rz(-1.9016409918667891) q[1];
ry(1.5717774087221583) q[2];
rz(0.3908584690445061) q[2];
ry(1.5838532317562386) q[3];
rz(3.1231612154749127) q[3];
ry(-2.144323717782901) q[4];
rz(-1.562917189786365) q[4];
ry(-2.0746771855237247) q[5];
rz(1.5572204388352455) q[5];
ry(0.48871883096613056) q[6];
rz(-3.138986468315279) q[6];
ry(1.5713040299423182) q[7];
rz(2.3498315100559957) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5634605705436666) q[0];
rz(-1.7730256449060091) q[0];
ry(-0.04981449499391388) q[1];
rz(1.8992648771850091) q[1];
ry(3.1397596582122023) q[2];
rz(-1.1799537570067313) q[2];
ry(-3.1005019016937365) q[3];
rz(3.118438824246867) q[3];
ry(1.7896392191271977) q[4];
rz(-1.6311443570996298) q[4];
ry(-1.0704556423088971) q[5];
rz(1.6338917166124518) q[5];
ry(2.7440704206163224) q[6];
rz(0.0009549749309413778) q[6];
ry(-0.0015275129067019136) q[7];
rz(2.114439704039427) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.1411260263748613) q[0];
rz(-1.02445132390322) q[0];
ry(1.5717781067022938) q[1];
rz(-1.174284690988019) q[1];
ry(-1.5700441813369919) q[2];
rz(-2.9817074321826937) q[2];
ry(1.5704141992959446) q[3];
rz(3.1412046376674994) q[3];
ry(-0.3315946112681898) q[4];
rz(0.07816865466883752) q[4];
ry(2.5777199533272355) q[5];
rz(-1.5855867354139441) q[5];
ry(-2.827402140855522) q[6];
rz(3.140140377319217) q[6];
ry(-0.0005404857477047685) q[7];
rz(-2.891278087360592) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.139173317288424) q[0];
rz(-0.8220528209912661) q[0];
ry(-3.141444739093065) q[1];
rz(0.3964261652060958) q[1];
ry(0.0021138082361273907) q[2];
rz(-1.7307730673180455) q[2];
ry(1.569814525716791) q[3];
rz(1.570727006730456) q[3];
ry(-1.5434717085177385) q[4];
rz(0.01680210891873024) q[4];
ry(1.5677085668448811) q[5];
rz(-3.125278166988376) q[5];
ry(2.4079685881930097) q[6];
rz(-1.5710181414865838) q[6];
ry(-1.5691843103685372) q[7];
rz(-3.1412767075868344) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.6310567137174772) q[0];
rz(0.002928517526582922) q[0];
ry(1.0526910858106946) q[1];
rz(3.1412939244859612) q[1];
ry(-1.5705087545835674) q[2];
rz(-0.36604912035304993) q[2];
ry(-1.547907271956861) q[3];
rz(0.00018631330919305353) q[3];
ry(1.0568142915781893) q[4];
rz(3.1277550895356496) q[4];
ry(-0.7592062515980116) q[5];
rz(0.8889395690185696) q[5];
ry(-1.5711261164941677) q[6];
rz(-3.141553296104233) q[6];
ry(1.5701948743094982) q[7];
rz(1.570475100893197) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5694995603215536) q[0];
rz(-3.0910054851393234) q[0];
ry(-1.5655570883699275) q[1];
rz(-1.833104506987592) q[1];
ry(-0.0003917116081737504) q[2];
rz(1.8336534604969001) q[2];
ry(1.5707325101633747) q[3];
rz(0.000742270181166127) q[3];
ry(-1.654482153125073) q[4];
rz(-1.6390767211316704) q[4];
ry(-0.09577874516015239) q[5];
rz(0.6738299023577882) q[5];
ry(-1.5710680906637808) q[6];
rz(-3.1397621336986576) q[6];
ry(1.4335591863178982) q[7];
rz(3.612139022290961e-05) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.024645618352461443) q[0];
rz(-2.5856034908431096) q[0];
ry(0.00013223095640831133) q[1];
rz(-0.12422312360188671) q[1];
ry(-0.011247545169362105) q[2];
rz(-2.3571890263116964) q[2];
ry(2.936445729343722) q[3];
rz(0.1603009165224929) q[3];
ry(-1.5699349835745169) q[4];
rz(0.008781651149351255) q[4];
ry(-1.570880200700783) q[5];
rz(-3.1403629352085014) q[5];
ry(1.5705803339367543) q[6];
rz(-1.56147229392149) q[6];
ry(1.582323126332315) q[7];
rz(1.5723329027953081) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.12988939057314) q[0];
rz(-2.666052135757943) q[0];
ry(-3.139930714851496) q[1];
rz(1.0458559367084534) q[1];
ry(3.131162432557556) q[2];
rz(-2.594651939800869) q[2];
ry(-3.1361260717579076) q[3];
rz(0.021907744117974914) q[3];
ry(1.5620102195734589) q[4];
rz(3.008268645456362) q[4];
ry(-1.566380936816297) q[5];
rz(-0.137421218726267) q[5];
ry(1.5798236396622116) q[6];
rz(1.4382847815648987) q[6];
ry(1.565550482026663) q[7];
rz(3.0040935981620023) q[7];