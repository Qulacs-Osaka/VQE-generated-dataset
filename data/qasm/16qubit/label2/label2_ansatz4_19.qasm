OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(0.0015629908799880809) q[0];
rz(2.22204710579257) q[0];
ry(3.137882858778914) q[1];
rz(0.7014988935434907) q[1];
ry(-1.5648929038791866) q[2];
rz(0.0002075281334797687) q[2];
ry(-2.98920090259385) q[3];
rz(-2.7549430777032984) q[3];
ry(-0.00024024512157172495) q[4];
rz(2.355023781310115) q[4];
ry(-3.141528475625924) q[5];
rz(-1.5624514270896033) q[5];
ry(-3.140000856309235) q[6];
rz(-0.21152406559239534) q[6];
ry(-3.1409533549707027) q[7];
rz(-2.382353644678131) q[7];
ry(-1.5702994251192752) q[8];
rz(1.712236097075349) q[8];
ry(-1.5711490498533083) q[9];
rz(-1.3468462297002743) q[9];
ry(3.1409464552637396) q[10];
rz(3.0542602945212556) q[10];
ry(-0.0003979630289716596) q[11];
rz(2.878321846180092) q[11];
ry(-1.5325429396980195) q[12];
rz(3.0728571125610644) q[12];
ry(-1.5686941819961) q[13];
rz(0.2595033476032711) q[13];
ry(3.1158933660204116) q[14];
rz(-0.01611070237719225) q[14];
ry(3.1288674851383385) q[15];
rz(0.574360082737788) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.57250601752826) q[0];
rz(-2.1897756587803485) q[0];
ry(1.5710359181753724) q[1];
rz(0.9004107103424515) q[1];
ry(1.5780470479356792) q[2];
rz(3.1073368369012404) q[2];
ry(0.0067463384380817) q[3];
rz(2.646553211466039) q[3];
ry(3.1368541333138094) q[4];
rz(-1.7988928001539248) q[4];
ry(-0.0013342461424885599) q[5];
rz(1.891753213655806) q[5];
ry(-1.597830121945214) q[6];
rz(-3.0304366741669595) q[6];
ry(1.6138774040223147) q[7];
rz(0.36428680501935506) q[7];
ry(-1.7935572179769248) q[8];
rz(-0.13140068101638483) q[8];
ry(-1.4334589990976083) q[9];
rz(1.3772472981154564) q[9];
ry(-0.6705686876158747) q[10];
rz(0.7849240360629118) q[10];
ry(0.9552536931295403) q[11];
rz(-2.2609323360647) q[11];
ry(1.6799826762981391) q[12];
rz(1.8515041486574446) q[12];
ry(0.13396148425117202) q[13];
rz(-2.162881162040435) q[13];
ry(-1.9096829745196091) q[14];
rz(2.9434903331457827) q[14];
ry(-1.514411064139467) q[15];
rz(-1.244667263695263) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.9982445267982607) q[0];
rz(-2.091779674649234) q[0];
ry(1.0519233182847485) q[1];
rz(-1.2779066562749781) q[1];
ry(-1.5754361579216105) q[2];
rz(-1.9009946281189603) q[2];
ry(1.5834511425806541) q[3];
rz(-3.001274394736887) q[3];
ry(-3.0839821814970323) q[4];
rz(1.038726299857613) q[4];
ry(-3.1195004746575483) q[5];
rz(-3.0901450450911647) q[5];
ry(1.8947499144037376) q[6];
rz(0.46173060546902706) q[6];
ry(2.7820082122534746) q[7];
rz(-0.14029906427339828) q[7];
ry(-0.9811946536913076) q[8];
rz(-3.1178351491009595) q[8];
ry(2.1611676611215183) q[9];
rz(-0.02926758415558961) q[9];
ry(1.2864269164973603) q[10];
rz(2.4781773566539624) q[10];
ry(1.7308688954649263) q[11];
rz(-2.015024262362486) q[11];
ry(1.7767960923405894) q[12];
rz(-2.2729024218703824) q[12];
ry(1.4142723501574574) q[13];
rz(-2.289167733971807) q[13];
ry(-1.179092218260549) q[14];
rz(-2.9836415779199883) q[14];
ry(-1.4171897856760438) q[15];
rz(0.24981114546099548) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.0253336948009197) q[0];
rz(-2.9008346648441656) q[0];
ry(-2.304739868030532) q[1];
rz(2.019566696246109) q[1];
ry(-2.993874770724963) q[2];
rz(0.3419226696600217) q[2];
ry(-1.4015135334243634) q[3];
rz(2.9746652354803116) q[3];
ry(-2.6046142535969454) q[4];
rz(2.6952879301856485) q[4];
ry(-3.120596309844555) q[5];
rz(2.4953238220373506) q[5];
ry(-0.6389992318102369) q[6];
rz(1.673323970365904) q[6];
ry(-0.655943787740362) q[7];
rz(1.5000323308318024) q[7];
ry(0.33831966944988956) q[8];
rz(1.8205664550799854) q[8];
ry(-2.8090590567411864) q[9];
rz(-1.3045409985157268) q[9];
ry(3.104906223719688) q[10];
rz(-1.0958492254367065) q[10];
ry(2.65745756930018) q[11];
rz(-1.6274192342074787) q[11];
ry(-0.6614607286262357) q[12];
rz(-1.0050060913061927) q[12];
ry(0.6563938647218359) q[13];
rz(-0.941673634244628) q[13];
ry(2.947653839475839) q[14];
rz(-0.22037177835142077) q[14];
ry(-0.13146676535752097) q[15];
rz(0.09461649972278606) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.08305247407696735) q[0];
rz(1.2247849789026146) q[0];
ry(-2.4717502058492444) q[1];
rz(-1.2020268017530849) q[1];
ry(1.8027995202441882) q[2];
rz(-0.019136925544505523) q[2];
ry(0.0981453338913163) q[3];
rz(-1.95726467717917) q[3];
ry(2.086588092958055) q[4];
rz(0.8427007847121086) q[4];
ry(1.4665662398783008) q[5];
rz(2.198634151958859) q[5];
ry(1.6642629951024752) q[6];
rz(-2.8009079101153094) q[6];
ry(1.657081185051345) q[7];
rz(0.9895011063766486) q[7];
ry(-2.587625703815503) q[8];
rz(2.088056415508083) q[8];
ry(-0.5415440098908553) q[9];
rz(1.4838210497378093) q[9];
ry(-1.0408362772633766) q[10];
rz(-1.7618638093682684) q[10];
ry(-0.7155769267469233) q[11];
rz(-1.5914094150250158) q[11];
ry(-1.899522132588732) q[12];
rz(3.024485213529696) q[12];
ry(-0.9794919835931069) q[13];
rz(0.28080278693169625) q[13];
ry(0.09646787821005898) q[14];
rz(2.6922066481035967) q[14];
ry(2.063283096453919) q[15];
rz(1.1873878521573569) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.0017023253292213075) q[0];
rz(1.1425321973338043) q[0];
ry(-0.03194869879950072) q[1];
rz(-2.4429867178904203) q[1];
ry(-0.011345690265508758) q[2];
rz(2.883939362088586) q[2];
ry(3.13944973376807) q[3];
rz(2.8126658900206176) q[3];
ry(3.140051836839099) q[4];
rz(-2.5790206842844903) q[4];
ry(0.0036933363651888646) q[5];
rz(-0.3526577709967622) q[5];
ry(0.018511118375734294) q[6];
rz(-1.4670111080436339) q[6];
ry(-3.090491934725829) q[7];
rz(2.8038044305974172) q[7];
ry(-3.113030908164767) q[8];
rz(-0.1269764818189258) q[8];
ry(-0.03246151290587207) q[9];
rz(0.8754681999139207) q[9];
ry(-2.990968695602703) q[10];
rz(-2.3603018061063197) q[10];
ry(0.06581210526007988) q[11];
rz(0.7266244060711072) q[11];
ry(3.098486728016732) q[12];
rz(2.9925467716371497) q[12];
ry(-3.1207148631392814) q[13];
rz(0.2390543083178018) q[13];
ry(0.09677236528326816) q[14];
rz(-2.7492657256237667) q[14];
ry(-3.0340174344623385) q[15];
rz(-0.8544302598151484) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.641440436906568) q[0];
rz(0.4731119634010072) q[0];
ry(-1.724471876767985) q[1];
rz(0.8585649854651858) q[1];
ry(2.0908568229970133) q[2];
rz(2.9321379186851404) q[2];
ry(-2.756313750921673) q[3];
rz(2.6122241001266397) q[3];
ry(-1.270400051035975) q[4];
rz(-2.0922257847609753) q[4];
ry(-1.1396377862863085) q[5];
rz(2.17414715869685) q[5];
ry(2.9628262342407354) q[6];
rz(1.8858700125768857) q[6];
ry(0.17206857490077598) q[7];
rz(1.4130907929554184) q[7];
ry(3.135526917574652) q[8];
rz(1.8475301821071564) q[8];
ry(-0.027616735028362184) q[9];
rz(-0.11027776697891412) q[9];
ry(1.4000934679219919) q[10];
rz(0.3333434936191589) q[10];
ry(2.132862511363233) q[11];
rz(-1.9571435799402481) q[11];
ry(-1.9509735767330936) q[12];
rz(2.4533312945913215) q[12];
ry(-0.8791922621972103) q[13];
rz(-0.5963384422778366) q[13];
ry(-0.407373810573195) q[14];
rz(0.8142780758578176) q[14];
ry(1.7003392083769047) q[15];
rz(-0.840069406130073) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.3624113602225316) q[0];
rz(-1.1477432812633672) q[0];
ry(0.9853727362229323) q[1];
rz(0.2897947405502057) q[1];
ry(-3.1415623176469096) q[2];
rz(2.437132587084883) q[2];
ry(0.0033648814757972953) q[3];
rz(-3.013771244527169) q[3];
ry(1.719225365636487) q[4];
rz(0.06533869361618459) q[4];
ry(-0.7119841833747911) q[5];
rz(-3.030798365349568) q[5];
ry(1.734282840595518) q[6];
rz(2.018205990177276) q[6];
ry(1.3883661557517248) q[7];
rz(0.11721378577043053) q[7];
ry(-1.9900049215501583) q[8];
rz(2.769586575768715) q[8];
ry(-1.135029854076049) q[9];
rz(2.773433274867795) q[9];
ry(0.5004334283965733) q[10];
rz(0.3055438665495939) q[10];
ry(1.4953784012676312) q[11];
rz(1.3477004473806802) q[11];
ry(-2.955932521858302) q[12];
rz(-2.397328326889642) q[12];
ry(0.18168139770067881) q[13];
rz(2.3671362048310876) q[13];
ry(1.1874423859418188) q[14];
rz(3.108143860468348) q[14];
ry(1.194786774905408) q[15];
rz(0.2050689864174187) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.6666815037588636) q[0];
rz(2.9511291034569704) q[0];
ry(1.91472049004278) q[1];
rz(0.38600794255502563) q[1];
ry(0.003494669459629622) q[2];
rz(-0.6340597694723956) q[2];
ry(-3.1363050763238696) q[3];
rz(-2.207293816582795) q[3];
ry(-1.4529587529180086) q[4];
rz(1.107603692148575) q[4];
ry(0.8764717120577918) q[5];
rz(-1.0963230828688448) q[5];
ry(-3.134796052023563) q[6];
rz(0.20646164494624486) q[6];
ry(3.137539481869973) q[7];
rz(1.408664829389543) q[7];
ry(-1.5429660161662788) q[8];
rz(0.815357541777735) q[8];
ry(1.5651504224198218) q[9];
rz(-2.0627970805557183) q[9];
ry(-3.0372390351661838) q[10];
rz(1.1368485001155515) q[10];
ry(3.0024254105024633) q[11];
rz(0.7238096018024551) q[11];
ry(0.2670025440942221) q[12];
rz(1.3415762493245822) q[12];
ry(2.8765814492147785) q[13];
rz(1.920664782377023) q[13];
ry(-1.703281568302467) q[14];
rz(-2.6626936469807356) q[14];
ry(0.43019313458246433) q[15];
rz(-0.16130675762470317) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.560753414450075) q[0];
rz(1.4927428356990058) q[0];
ry(-1.4651380250960884) q[1];
rz(-0.6810572375988343) q[1];
ry(-0.00846199335797891) q[2];
rz(2.0718605899425) q[2];
ry(-3.1339898147021765) q[3];
rz(-0.2335788421452654) q[3];
ry(-2.226549992326587) q[4];
rz(-0.5990359759092643) q[4];
ry(2.0546013671491474) q[5];
rz(-2.7682579648180305) q[5];
ry(1.9118572902675623) q[6];
rz(-1.3037628363393066) q[6];
ry(-1.22051912078296) q[7];
rz(1.3329929929735522) q[7];
ry(-3.08808789400224) q[8];
rz(-0.17518427777548906) q[8];
ry(-3.093360320327685) q[9];
rz(-1.5534286943964235) q[9];
ry(-1.6644183991169168) q[10];
rz(1.4133492905640928) q[10];
ry(2.4342234791556723) q[11];
rz(0.5143882062109988) q[11];
ry(-0.012911232155093265) q[12];
rz(-1.07037368450219) q[12];
ry(-3.115675894358601) q[13];
rz(1.6277352087775858) q[13];
ry(0.6097787137800854) q[14];
rz(-0.7324823705836215) q[14];
ry(-2.0865070894452726) q[15];
rz(-0.4741148330523029) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.2024154936189437) q[0];
rz(-2.506423083740013) q[0];
ry(-2.9812836778793175) q[1];
rz(0.6052766733201173) q[1];
ry(0.05415478283062661) q[2];
rz(2.009130577591012) q[2];
ry(3.1305976121659853) q[3];
rz(-2.730171055125967) q[3];
ry(1.2972303235735918) q[4];
rz(-1.0975372702765256) q[4];
ry(1.2789184914066443) q[5];
rz(2.9026317887565067) q[5];
ry(-1.427412298731523) q[6];
rz(-1.1272626635543181) q[6];
ry(1.4447032616183124) q[7];
rz(2.298537392314289) q[7];
ry(-0.7302102007695973) q[8];
rz(-2.1421190234874614) q[8];
ry(0.11396886644387738) q[9];
rz(2.7970868395923336) q[9];
ry(-2.141461046661592) q[10];
rz(-1.4872888315452004) q[10];
ry(-2.2094502870598203) q[11];
rz(-1.983462186542373) q[11];
ry(2.415248875722051) q[12];
rz(-1.28243654294844) q[12];
ry(2.4815061005333874) q[13];
rz(1.3707148502592748) q[13];
ry(-2.185442948689472) q[14];
rz(-2.5982567779242896) q[14];
ry(-2.3254907945020444) q[15];
rz(-3.0326435623993415) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.8338942013079115) q[0];
rz(-0.9450959510951192) q[0];
ry(2.387756600362713) q[1];
rz(0.5967784531624688) q[1];
ry(-2.049244695856908) q[2];
rz(0.009792414490544132) q[2];
ry(2.9396050394339497) q[3];
rz(0.637420790673473) q[3];
ry(1.8649933049688074) q[4];
rz(2.237224018067005) q[4];
ry(1.849412227792612) q[5];
rz(2.201442736353365) q[5];
ry(-1.2739540207833837) q[6];
rz(3.1141563969392836) q[6];
ry(3.122045183455618) q[7];
rz(-0.481941582598946) q[7];
ry(-0.16494746960066947) q[8];
rz(2.9754517680791985) q[8];
ry(0.12949754389296567) q[9];
rz(0.1966044670029543) q[9];
ry(-0.008049105177312561) q[10];
rz(-2.283574800029015) q[10];
ry(3.1411955338818855) q[11];
rz(1.058220659576765) q[11];
ry(0.1597968010752341) q[12];
rz(-2.1272421921609546) q[12];
ry(-0.18644410159977998) q[13];
rz(-1.2511497996581369) q[13];
ry(-3.128721562227973) q[14];
rz(-1.753286587259935) q[14];
ry(-2.9009653853533326) q[15];
rz(-0.5201835502691513) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.091661241579585) q[0];
rz(-2.3804765256673095) q[0];
ry(-3.133012271027364) q[1];
rz(1.7099867802174682) q[1];
ry(-0.0029907559681028317) q[2];
rz(-2.0758872650823688) q[2];
ry(-3.139616250970513) q[3];
rz(1.6022423454496364) q[3];
ry(-3.1410800181089478) q[4];
rz(-2.728513829889726) q[4];
ry(-3.1415914824794866) q[5];
rz(-2.5331281369742444) q[5];
ry(-3.135418734308912) q[6];
rz(3.0822767044624295) q[6];
ry(-0.0011823613005175204) q[7];
rz(1.2640067642127804) q[7];
ry(3.0816801120444923) q[8];
rz(2.5899276838914504) q[8];
ry(3.067123873550356) q[9];
rz(-0.09892989435039201) q[9];
ry(2.801624659533806) q[10];
rz(-2.5506128522519895) q[10];
ry(-0.2868633831169417) q[11];
rz(1.38654935766857) q[11];
ry(-0.241605718688184) q[12];
rz(-2.898516847339156) q[12];
ry(0.2634389282023246) q[13];
rz(3.0406732338310842) q[13];
ry(-2.052351132171536) q[14];
rz(-2.2473456040174833) q[14];
ry(2.2922572688854586) q[15];
rz(0.4151337743939058) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.9615679340588037) q[0];
rz(2.7812580710882884) q[0];
ry(-3.0795012353111204) q[1];
rz(-0.4089350318195519) q[1];
ry(2.204686302420032) q[2];
rz(-0.5235433444251589) q[2];
ry(0.9615023929827062) q[3];
rz(-0.9413119771400418) q[3];
ry(-1.5243414219140679) q[4];
rz(-0.3756999051048968) q[4];
ry(2.3546303608746464) q[5];
rz(1.240530256783933) q[5];
ry(-1.9080350477927421) q[6];
rz(0.9779033699961658) q[6];
ry(-3.1282673126380067) q[7];
rz(1.9672609183621834) q[7];
ry(-1.6556693496054633) q[8];
rz(2.949509659848196) q[8];
ry(-1.5410165203473047) q[9];
rz(1.4962190802644368) q[9];
ry(-3.1303243491797117) q[10];
rz(-1.4163840444030704) q[10];
ry(0.008975187210982227) q[11];
rz(-0.16588684765926234) q[11];
ry(-1.8224890576581765) q[12];
rz(-1.1606585487073153) q[12];
ry(-1.7903160794721937) q[13];
rz(1.1559585816909372) q[13];
ry(2.4405942503538123) q[14];
rz(-2.317787785886237) q[14];
ry(-2.6578440823235137) q[15];
rz(0.458386956686107) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.19696695844794163) q[0];
rz(-0.5049852171795506) q[0];
ry(2.5704468386468116) q[1];
rz(-0.24669290246234077) q[1];
ry(0.016651733835093552) q[2];
rz(-2.960640886752167) q[2];
ry(3.1239248067204675) q[3];
rz(1.2546996170874578) q[3];
ry(-0.8477766460731821) q[4];
rz(2.2124863675919304) q[4];
ry(-0.20166937402050067) q[5];
rz(1.8419429516544912) q[5];
ry(0.853152515726908) q[6];
rz(1.4464332469867336) q[6];
ry(1.492420465733013) q[7];
rz(0.25498254686955146) q[7];
ry(-3.055573575496888) q[8];
rz(-0.4352764180719113) q[8];
ry(-0.7594563405601147) q[9];
rz(1.1785964635737964) q[9];
ry(-0.08382570921691368) q[10];
rz(-1.3331162686752485) q[10];
ry(3.0262578938616214) q[11];
rz(-2.6796725062136373) q[11];
ry(-1.6388843529320551) q[12];
rz(0.33754198539698826) q[12];
ry(1.629237697386082) q[13];
rz(2.9549864986289394) q[13];
ry(-0.9158621780764128) q[14];
rz(-0.04035739905273683) q[14];
ry(-2.9394618284123157) q[15];
rz(0.9858990083928535) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(3.1414734121010133) q[0];
rz(1.0911322635480818) q[0];
ry(0.288832597393491) q[1];
rz(2.964229982471591) q[1];
ry(-3.1287389252954485) q[2];
rz(1.2133129308274224) q[2];
ry(0.015269299471651522) q[3];
rz(0.25512264875616) q[3];
ry(0.0037246923877182875) q[4];
rz(-2.1570087636320467) q[4];
ry(-3.1350173711711014) q[5];
rz(0.7858782085790709) q[5];
ry(-0.001111819473559983) q[6];
rz(-3.023891498830016) q[6];
ry(0.006221489413330809) q[7];
rz(2.821985519350934) q[7];
ry(0.01330097503377825) q[8];
rz(2.9816600741512604) q[8];
ry(0.024518551210189088) q[9];
rz(-0.8204119441747046) q[9];
ry(2.568744335607067) q[10];
rz(2.508307923928955) q[10];
ry(1.8093264344884177) q[11];
rz(0.26782558198589695) q[11];
ry(-0.7357964739834091) q[12];
rz(0.7637920634875702) q[12];
ry(-2.3492573430315677) q[13];
rz(0.8146657667223128) q[13];
ry(1.2044570009299802) q[14];
rz(2.9060688487619215) q[14];
ry(0.5726310942034667) q[15];
rz(-1.1437244848158197) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.80904600200068) q[0];
rz(1.15084026259399) q[0];
ry(1.8072186546887445) q[1];
rz(-0.5703925495993264) q[1];
ry(2.7480674541089263) q[2];
rz(-1.638885402073722) q[2];
ry(2.7156231893437606) q[3];
rz(2.1632219134406627) q[3];
ry(1.0974049339621539) q[4];
rz(-1.8148052664504712) q[4];
ry(1.4481098510517707) q[5];
rz(2.07149502947075) q[5];
ry(-2.309722050702628) q[6];
rz(-0.6050324249923202) q[6];
ry(2.516888064196119) q[7];
rz(2.322613671061869) q[7];
ry(-0.04926864256761343) q[8];
rz(0.19996129730737974) q[8];
ry(0.010807194292094187) q[9];
rz(1.0391611112885963) q[9];
ry(-0.11246553759370592) q[10];
rz(-2.316251880931056) q[10];
ry(3.0287601110588684) q[11];
rz(-2.4184091982537805) q[11];
ry(-2.656531253820519) q[12];
rz(1.687818762358316) q[12];
ry(-0.5213338843337665) q[13];
rz(1.4818840276489356) q[13];
ry(0.4020007987368732) q[14];
rz(-0.4566823622687472) q[14];
ry(-3.122564613494496) q[15];
rz(-2.8440123008201734) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.1905624594785786) q[0];
rz(1.5944055723532484) q[0];
ry(1.9925427914892953) q[1];
rz(1.8750348988401009) q[1];
ry(3.1383715553150364) q[2];
rz(3.040637420728785) q[2];
ry(0.04606457441779713) q[3];
rz(-1.5656358231497027) q[3];
ry(0.0005134600176574428) q[4];
rz(-0.05362333733160619) q[4];
ry(-3.1379514362008343) q[5];
rz(-1.3869062923971662) q[5];
ry(3.1369746892822317) q[6];
rz(-1.1044679135157918) q[6];
ry(0.00033645260216537545) q[7];
rz(-2.780294511524878) q[7];
ry(-3.0627026715243972) q[8];
rz(-0.14647735681016805) q[8];
ry(3.1356303912904684) q[9];
rz(2.822983622861036) q[9];
ry(3.100361093734643) q[10];
rz(-1.677743783236164) q[10];
ry(0.3572423312015678) q[11];
rz(-1.4662996628072777) q[11];
ry(2.7750100215506004) q[12];
rz(0.9326292275712847) q[12];
ry(0.20717737733882746) q[13];
rz(-1.7691162193233927) q[13];
ry(0.6263566861035184) q[14];
rz(0.9418418700894349) q[14];
ry(-0.13021053322635787) q[15];
rz(-1.52425102735888) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.089771885255298) q[0];
rz(0.12201926189934921) q[0];
ry(0.007496611275758225) q[1];
rz(3.0487331287470614) q[1];
ry(1.8118331026500454) q[2];
rz(2.9041755864560903) q[2];
ry(1.0675549340998725) q[3];
rz(-0.03033700543888122) q[3];
ry(-0.8381658396640095) q[4];
rz(-1.9684780703281728) q[4];
ry(-2.353641103938019) q[5];
rz(-0.9287997056748473) q[5];
ry(2.4007168332843074) q[6];
rz(2.5263602702025785) q[6];
ry(-2.7430926878303326) q[7];
rz(1.8300340747560992) q[7];
ry(-1.6889579020177203) q[8];
rz(-3.003692373816014) q[8];
ry(2.779525394386714) q[9];
rz(-0.057844123079598006) q[9];
ry(-5.103013855922711e-06) q[10];
rz(-2.0991439265011156) q[10];
ry(0.011540968596522207) q[11];
rz(2.6662039308858296) q[11];
ry(-0.2966538559042301) q[12];
rz(0.37300874199214284) q[12];
ry(3.0082575926818578) q[13];
rz(2.4074496408059183) q[13];
ry(2.212009880180191) q[14];
rz(3.0062626551702007) q[14];
ry(-0.636549226122836) q[15];
rz(-2.50352861348783) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.955717744157728) q[0];
rz(-0.023395458898716193) q[0];
ry(-2.27885837384363) q[1];
rz(0.15429376549815288) q[1];
ry(-1.7919011120937904) q[2];
rz(-2.4234738533670614) q[2];
ry(-1.7677994812521474) q[3];
rz(-0.7892620740526324) q[3];
ry(3.134387309098322) q[4];
rz(2.1282972297104776) q[4];
ry(-0.006899207155304059) q[5];
rz(-1.5110221258541099) q[5];
ry(-0.0008511919780076483) q[6];
rz(-2.191902379959226) q[6];
ry(6.791183999510508e-05) q[7];
rz(-0.9462561624888384) q[7];
ry(3.0841714481377682) q[8];
rz(-2.9404252587089212) q[8];
ry(-3.112277089563589) q[9];
rz(0.6482743790719558) q[9];
ry(3.130360195682265) q[10];
rz(0.3406794618626022) q[10];
ry(0.008651997016782431) q[11];
rz(-0.2997477131658499) q[11];
ry(1.843867938261341) q[12];
rz(0.9198543508460137) q[12];
ry(2.226872355443261) q[13];
rz(2.2112895507940555) q[13];
ry(-0.0025671352908473804) q[14];
rz(2.3611130550853248) q[14];
ry(-0.3900721176480024) q[15];
rz(-1.4495085453256182) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.7846145494084045) q[0];
rz(-1.5503036049169656) q[0];
ry(3.1029373919609737) q[1];
rz(-1.4481240003251903) q[1];
ry(-3.0747616036136356) q[2];
rz(1.0731683960052152) q[2];
ry(-3.114704687108414) q[3];
rz(0.3844950702224752) q[3];
ry(-2.6047479293748204) q[4];
rz(-2.3033391718771656) q[4];
ry(-0.3567357124313961) q[5];
rz(0.10331291533348796) q[5];
ry(-2.9991020034973186) q[6];
rz(-1.8492863208348753) q[6];
ry(-1.5366490120065268) q[7];
rz(0.651093848686808) q[7];
ry(1.0984989175622522) q[8];
rz(-1.5789001711625799) q[8];
ry(-0.22388465106832633) q[9];
rz(-2.130179212078266) q[9];
ry(-3.128067984485677) q[10];
rz(0.8140829606898466) q[10];
ry(-3.1372788193747394) q[11];
rz(-1.2177056641380157) q[11];
ry(-2.4415541113135735) q[12];
rz(-1.5587900427301251) q[12];
ry(-1.0989059128478331) q[13];
rz(0.8781094490329399) q[13];
ry(-0.44922001808492595) q[14];
rz(2.7106363024417925) q[14];
ry(0.43281471429452145) q[15];
rz(2.772926563929659) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5467157840908834) q[0];
rz(-2.6557084354018223) q[0];
ry(1.57325865893939) q[1];
rz(2.6872399054113596) q[1];
ry(-3.136490080325969) q[2];
rz(2.8740722808789148) q[2];
ry(3.1381822902894188) q[3];
rz(-0.5968853076230273) q[3];
ry(-0.0007991072859656611) q[4];
rz(1.7809940827694746) q[4];
ry(0.007134263809801908) q[5];
rz(-2.464199113746417) q[5];
ry(3.1034541796260435) q[6];
rz(1.746467876560101) q[6];
ry(3.1094056629968643) q[7];
rz(1.530569322503579) q[7];
ry(-1.5476506565255352) q[8];
rz(-3.109005783589768) q[8];
ry(-1.5444985905917135) q[9];
rz(0.20006107190526468) q[9];
ry(0.00552482095734952) q[10];
rz(-0.9023471766579857) q[10];
ry(-3.135906367734134) q[11];
rz(-2.957291911830096) q[11];
ry(-1.061577430707886) q[12];
rz(-0.627308974533495) q[12];
ry(2.121459645433852) q[13];
rz(-3.0708961658533744) q[13];
ry(1.6444044596547764) q[14];
rz(-1.7697844724005227) q[14];
ry(1.495039684170744) q[15];
rz(-1.3548064319411628) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.4881542009565882) q[0];
rz(-0.03095229703957347) q[0];
ry(2.683745380728365) q[1];
rz(-2.132988404523248) q[1];
ry(3.1324729157114484) q[2];
rz(-2.5586729745988697) q[2];
ry(-2.746934448406683) q[3];
rz(2.6342985947054496) q[3];
ry(1.3061153299342443) q[4];
rz(2.2378626156340573) q[4];
ry(-1.7275992887762808) q[5];
rz(2.273986679989811) q[5];
ry(-2.2400125399846713) q[6];
rz(0.4758386986645587) q[6];
ry(0.9225294641271248) q[7];
rz(0.4268583185683337) q[7];
ry(-3.118889079451739) q[8];
rz(-1.9468692635006564) q[8];
ry(0.022031001669135467) q[9];
rz(-2.1642365853181156) q[9];
ry(-0.03832046863706129) q[10];
rz(-1.3179965646254226) q[10];
ry(-3.1321915984861306) q[11];
rz(-2.968779934764241) q[11];
ry(0.8992452891155134) q[12];
rz(-1.6591313737086768) q[12];
ry(0.6422525387619364) q[13];
rz(2.3709770163928137) q[13];
ry(2.8057543490904178) q[14];
rz(0.09729540538513426) q[14];
ry(0.3387568520081494) q[15];
rz(-3.0489842644329506) q[15];