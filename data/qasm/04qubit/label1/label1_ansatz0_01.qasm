OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.022725147485469242) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.06515389968085064) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.012339282893399472) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9482346823313508) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.9058314644188916) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.036597637732544666) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.00010594910674642719) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.5388940096631385) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.2940058229048655) q[3];
cx q[2],q[3];
rz(-0.07946648776412638) q[0];
rz(0.27691580290573325) q[1];
rz(0.6047650968445004) q[2];
rz(-0.4658207447258063) q[3];
rx(-0.5581564300230817) q[0];
rx(0.08911935681487762) q[1];
rx(-0.9118377184943981) q[2];
rx(-1.0620180808799833) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.04465604174876796) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.2952721282098711) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.13214622418279628) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.3124264982343468) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.9467276528392039) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15854448780918293) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.7175942102420497) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.001983547813488674) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.5447523389861616) q[3];
cx q[2],q[3];
rz(0.06795528176276583) q[0];
rz(0.28110920908229897) q[1];
rz(0.2161367289796094) q[2];
rz(-0.4542008023304777) q[3];
rx(-0.7561237632358982) q[0];
rx(-0.25251666298247955) q[1];
rx(-1.1778658652039562) q[2];
rx(-1.0043167704292957) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.06343938250997945) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.545635206646751) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.06096666913563659) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.41382848875508) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.5954785968808084) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2637866677367762) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.6797383045580373) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.35478195537065466) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.10509983202065354) q[3];
cx q[2],q[3];
rz(0.3815762095563937) q[0];
rz(0.23720949506725247) q[1];
rz(0.5750905838140027) q[2];
rz(0.39805673856849766) q[3];
rx(-0.8161957413916529) q[0];
rx(-0.11191549556262545) q[1];
rx(0.037149064776800134) q[2];
rx(-1.0658330479642484) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.33110005139654536) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.6239740207420138) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.12070446575902952) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9206058632400086) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.3294698147218205) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.34704330377820647) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.269640333353855) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.4410670131898639) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.021579814233723283) q[3];
cx q[2],q[3];
rz(-0.3167572844432695) q[0];
rz(-0.23931347695443012) q[1];
rz(0.12226917688321252) q[2];
rz(0.30561525522229516) q[3];
rx(-1.0279683182874888) q[0];
rx(-0.10468089716080843) q[1];
rx(0.3114280167532927) q[2];
rx(-0.8342511258450984) q[3];