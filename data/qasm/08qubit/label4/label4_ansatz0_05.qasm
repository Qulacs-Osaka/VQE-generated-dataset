OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.3303747908023602) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.15639744091221106) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.2618026856525623) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.033408235100206546) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0321521077410088) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-2.0080264839823206e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.0008372857944651372) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.09504909153974343) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.2840770709937577) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.28788463600986) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.1564656135720583) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.08925166979241558) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.11307326628895495) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.24293429010446446) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.4308087205602897) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.2353916412662073) q[7];
cx q[6],q[7];
rx(-0.3967316523076473) q[0];
rz(-0.09499937846815593) q[0];
rx(-0.020082971577871523) q[1];
rz(-0.2766021594576373) q[1];
rx(-0.06673436110777581) q[2];
rz(-0.2791308698121927) q[2];
rx(0.06826564541710736) q[3];
rz(-0.11111720998401445) q[3];
rx(0.022332529917522125) q[4];
rz(0.05468877744518774) q[4];
rx(-0.01094682189249769) q[5];
rz(-0.40256195327880245) q[5];
rx(-0.01092725054402687) q[6];
rz(-0.06975635942488559) q[6];
rx(-0.13791079018654787) q[7];
rz(-0.1879824137394035) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2550362297042896) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.13547167477045774) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.19349540844054214) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.39847497718260794) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.15900474507066806) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(9.169236235565013e-05) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.00031514883442065346) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.283337178775423) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.001010687803617554) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.0024689372121266266) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.3257914351236704) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.4950545391538115) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.008325741766391892) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.08676809875691832) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.29426361309005583) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.0798248979063935) q[7];
cx q[6],q[7];
rx(-0.5789198094524072) q[0];
rz(0.07618644286455019) q[0];
rx(0.02442074873315359) q[1];
rz(-0.29303157627752024) q[1];
rx(-0.036085995506542116) q[2];
rz(-0.25697758916489777) q[2];
rx(-0.39345416601814437) q[3];
rz(-0.28434681790245314) q[3];
rx(-0.5831196486857624) q[4];
rz(0.09096553220201078) q[4];
rx(0.03663619946924103) q[5];
rz(-0.390128890411732) q[5];
rx(0.01959521037808703) q[6];
rz(-0.1260756072449861) q[6];
rx(-0.3716761346556866) q[7];
rz(-0.1855480423254928) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.406561926531178) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03693355338286995) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08691024255697086) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.20319358587423453) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.23576333463760113) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.00029176442554872475) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-9.675273142878187e-07) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.08707038948639195) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-3.379591436974451e-06) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.0006104221151195801) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.31152758569712563) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.013504089266572242) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.011784476130587604) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.1447772521077393) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.3535459289512931) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.1552320723749084) q[7];
cx q[6],q[7];
rx(-0.6654558207037232) q[0];
rz(0.013754246199886825) q[0];
rx(-0.02913983729225376) q[1];
rz(-0.18490742989639042) q[1];
rx(0.027832636828766143) q[2];
rz(-0.2398123897492551) q[2];
rx(-0.9185450275040957) q[3];
rz(-0.02413010074118316) q[3];
rx(-1.1065703789310066) q[4];
rz(-0.09111439682264198) q[4];
rx(-0.04388570736612638) q[5];
rz(-0.15820482399140198) q[5];
rx(0.043067290873853525) q[6];
rz(-0.12467172823772034) q[6];
rx(-0.5031469116725987) q[7];
rz(-0.006276400173472537) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.44427391679107564) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03445573264100998) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.03802645388860998) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05597487400902332) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0426184868428155) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.0003268380665956465) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-4.35738300641835e-06) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.00754096685007306) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(9.696997252094827e-05) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.0002390524814705923) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.2906363111129668) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.0010165014461163088) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.03539682508665238) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.36271529921982326) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(0.26929522019617186) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.5043105843183762) q[7];
cx q[6],q[7];
rx(-0.544347706315903) q[0];
rz(-0.0007144392244759623) q[0];
rx(0.02248603571997843) q[1];
rz(-0.36110153024319663) q[1];
rx(-0.08295105143457472) q[2];
rz(-0.20959413480353234) q[2];
rx(-1.1511801353011928) q[3];
rz(0.14336098281342227) q[3];
rx(-1.0212162612410665) q[4];
rz(0.12889913329918365) q[4];
rx(-0.08080335880820196) q[5];
rz(-0.03329924139365408) q[5];
rx(0.01933008715349721) q[6];
rz(-0.3115826884160965) q[6];
rx(-1.3156065129588534) q[7];
rz(-0.07624871080001266) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.33124371323429264) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.014536017487548476) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08615115647325032) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.1888847986115729) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04935011682939253) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.0003101182112756714) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(4.696084457674975e-06) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.012392051918812234) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.0004913831272109022) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.0005789608367558722) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.10103162205821556) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(0.014002431621725441) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.05404435747873335) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.11027757039133108) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.511394821835362) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.505565725429153) q[7];
cx q[6],q[7];
rx(-0.7504236147863393) q[0];
rz(-0.055585485719507076) q[0];
rx(-0.011242888545498122) q[1];
rz(-0.48935747082279324) q[1];
rx(0.0584695942955698) q[2];
rz(-0.29287996051195014) q[2];
rx(-0.7638554934545462) q[3];
rz(-0.2588951807455619) q[3];
rx(-0.4526959121920376) q[4];
rz(0.3770824934601191) q[4];
rx(0.26827688241004005) q[5];
rz(-0.5309164722445817) q[5];
rx(-0.0032553880181796814) q[6];
rz(-0.7196028281995429) q[6];
rx(-0.5773845958616589) q[7];
rz(0.35863591518917626) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12976135025833913) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.25436572662310253) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.3089822492597155) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.6312274126271883) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.8173141207641331) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.25943628683564157) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.2601781357858112) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.022238195297699697) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.21415718584809829) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.21379063425177575) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.2357226894225863) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.22828068390327097) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.38205201087484963) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.00376993241362367) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.0027755341679362427) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.0029278931520241042) q[7];
cx q[6],q[7];
rx(-0.2387020527484029) q[0];
rz(0.028616303209024998) q[0];
rx(0.0022098968443471983) q[1];
rz(-0.7802744352530931) q[1];
rx(-0.0008021639404402686) q[2];
rz(-0.04245226938338001) q[2];
rx(-8.09711403066255e-05) q[3];
rz(-0.428863080089966) q[3];
rx(-0.00021365151184986626) q[4];
rz(-0.047189154324918714) q[4];
rx(-0.0006427842817737809) q[5];
rz(-0.43507492439857864) q[5];
rx(-0.0005494610860806268) q[6];
rz(-0.04078663758743153) q[6];
rx(-0.5879445817973076) q[7];
rz(0.671574732580745) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08442935941096903) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08274281736675848) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.423168818614371) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.00922028994372175) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0106623147640648) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.809533767390112) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(-0.8104019978645425) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(-0.0024229516337317286) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.7753554427674899) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(-0.7755382366482219) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.8110475183843557) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.8093350487703217) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(-0.01690437703769146) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.6407570609382391) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.6412842950004681) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.3959826741453265) q[7];
cx q[6],q[7];
rx(0.00847376250238319) q[0];
rz(-0.011798398068235662) q[0];
rx(0.0008460563954822697) q[1];
rz(-0.40674886302919) q[1];
rx(0.00048686314142042965) q[2];
rz(-0.009190158331917138) q[2];
rx(-9.331931847748175e-05) q[3];
rz(-0.13663804572506236) q[3];
rx(-0.00015065621638169426) q[4];
rz(-0.01872713380855449) q[4];
rx(0.0001259044506918301) q[5];
rz(-0.1516284481554082) q[5];
rx(-2.224214205905236e-05) q[6];
rz(-0.02229889601888358) q[6];
rx(-0.0011203355925414773) q[7];
rz(-0.15747073839554498) q[7];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.027270502647014996) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0275033797197132) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.30911502116563827) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.01323061605575627) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.01290390931874518) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
h q[2];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.181789500971222) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
h q[4];
sdg q[2];
h q[2];
sdg q[4];
h q[4];
cx q[2],q[3];
cx q[3],q[4];
rz(0.18334803498005428) q[4];
cx q[3],q[4];
cx q[2],q[3];
h q[2];
s q[2];
h q[4];
s q[4];
cx q[2],q[3];
rz(0.04854790049245113) q[3];
cx q[2],q[3];
h q[3];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.17417465813347316) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
h q[5];
sdg q[3];
h q[3];
sdg q[5];
h q[5];
cx q[3],q[4];
cx q[4],q[5];
rz(0.17479201660517124) q[5];
cx q[4],q[5];
cx q[3],q[4];
h q[3];
s q[3];
h q[5];
s q[5];
h q[4];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.7166740485583123) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
h q[6];
sdg q[4];
h q[4];
sdg q[6];
h q[6];
cx q[4],q[5];
cx q[5],q[6];
rz(-0.7168566689511694) q[6];
cx q[5],q[6];
cx q[4],q[5];
h q[4];
s q[4];
h q[6];
s q[6];
cx q[4],q[5];
rz(0.007238026291366698) q[5];
cx q[4],q[5];
h q[5];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.7060229271959645) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
h q[7];
sdg q[5];
h q[5];
sdg q[7];
h q[7];
cx q[5],q[6];
cx q[6],q[7];
rz(-0.7055197870065949) q[7];
cx q[6],q[7];
cx q[5],q[6];
h q[5];
s q[5];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.031736259756987) q[7];
cx q[6],q[7];
rx(0.021622378817993865) q[0];
rz(-0.04965445251748173) q[0];
rx(-0.0004127143407651199) q[1];
rz(-0.1543961764358571) q[1];
rx(-0.001027037860402398) q[2];
rz(-0.06123008967251706) q[2];
rx(0.0001627144618426119) q[3];
rz(-0.03482272770886565) q[3];
rx(0.00010913393369404736) q[4];
rz(-0.051762043330937625) q[4];
rx(-0.0002794949675181908) q[5];
rz(-0.016284451726382147) q[5];
rx(-4.343925614601404e-05) q[6];
rz(-0.049840940110222026) q[6];
rx(0.001310892770520277) q[7];
rz(-0.016635687891060503) q[7];