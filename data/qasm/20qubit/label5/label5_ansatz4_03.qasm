OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.0738723964549646) q[0];
rz(1.521427314070019) q[0];
ry(-0.0004906208952614275) q[1];
rz(-1.4784853192132943) q[1];
ry(1.5708738932544533) q[2];
rz(-0.5772557098463635) q[2];
ry(1.561214198841614) q[3];
rz(-0.03117998237001443) q[3];
ry(-1.6231992243496771) q[4];
rz(-1.5813720766294759) q[4];
ry(-0.034384784945664604) q[5];
rz(1.5730883235223327) q[5];
ry(3.141573238203064) q[6];
rz(1.19223417155089) q[6];
ry(-3.1414234944493686) q[7];
rz(-0.5677167561130592) q[7];
ry(3.0621938217496267) q[8];
rz(2.9195574953415204) q[8];
ry(0.04323370394663062) q[9];
rz(-1.5698058536719395) q[9];
ry(1.5707954851018033) q[10];
rz(-1.5317152177190636) q[10];
ry(1.5707973180019168) q[11];
rz(-0.010167162619135086) q[11];
ry(-1.5707947572972358) q[12];
rz(0.0005510502515519988) q[12];
ry(1.5707954306018133) q[13];
rz(1.574689186047502) q[13];
ry(3.117607293551276) q[14];
rz(-2.853109061888803) q[14];
ry(-2.7092589920940378) q[15];
rz(-1.5686870203342191) q[15];
ry(-3.824709182254082e-06) q[16];
rz(-1.0866611255156577) q[16];
ry(-1.571534737948836) q[17];
rz(-0.000531122589273016) q[17];
ry(3.139980830529613) q[18];
rz(1.6196215028223027) q[18];
ry(-3.1414829216951246) q[19];
rz(0.9802660858689661) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5837998425594373) q[0];
rz(0.24652539550314412) q[0];
ry(3.0920566789228574) q[1];
rz(-0.48036989770948146) q[1];
ry(2.1702053984498506) q[2];
rz(0.41045757843576053) q[2];
ry(1.498324368878439) q[3];
rz(3.060935246733923) q[3];
ry(-1.5863248769572083) q[4];
rz(0.6778040115702696) q[4];
ry(1.570851323874055) q[5];
rz(-1.5587798688073873) q[5];
ry(-3.1415919524493012) q[6];
rz(-2.089090903201404) q[6];
ry(-3.1415703659631653) q[7];
rz(-2.2599418640650053) q[7];
ry(-3.1409842097790914) q[8];
rz(1.3484879976308113) q[8];
ry(1.570816360402163) q[9];
rz(0.8872831272203354) q[9];
ry(1.445236129929957) q[10];
rz(-0.30163678808851097) q[10];
ry(2.864081687044787) q[11];
rz(-3.0971579845090074) q[11];
ry(2.6828936046048346) q[12];
rz(3.1267992653981236) q[12];
ry(-1.5619142983582686) q[13];
rz(2.032642736622134) q[13];
ry(3.1411149046491453) q[14];
rz(-2.8541357222022605) q[14];
ry(-1.5707801649030946) q[15];
rz(-0.0019556062736059447) q[15];
ry(-1.5707961196157039) q[16];
rz(-2.0774894539577327) q[16];
ry(-1.5693848768449863) q[17];
rz(-0.0020657545699215023) q[17];
ry(-3.141090534744094) q[18];
rz(-1.7376843354385851) q[18];
ry(1.5707944772024318) q[19];
rz(-1.568774640042905) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5645384852653557) q[0];
rz(0.01538325638069627) q[0];
ry(-1.5755012105750028) q[1];
rz(-1.464656266391862) q[1];
ry(-0.0009909312165374828) q[2];
rz(-0.8293816858765339) q[2];
ry(-0.02129299243159273) q[3];
rz(3.0074442342081493) q[3];
ry(0.023148577562763784) q[4];
rz(-0.022342383088174187) q[4];
ry(1.6789785865383335) q[5];
rz(-1.5709413094964622) q[5];
ry(1.5708117348177373) q[6];
rz(0.3018579562936123) q[6];
ry(-2.056423917764961) q[7];
rz(-0.09402649164720081) q[7];
ry(1.5753396078280737) q[8];
rz(-0.004797955199964238) q[8];
ry(4.697193588394453e-06) q[9];
rz(0.6809961997425299) q[9];
ry(3.1241525704360695) q[10];
rz(-1.5941729605058983) q[10];
ry(-3.1213330068560845) q[11];
rz(-3.0886576545432836) q[11];
ry(1.2498465146231714) q[12];
rz(2.782449700132638) q[12];
ry(1.2090886006559516) q[13];
rz(1.5686005540196308) q[13];
ry(1.5707908035315423) q[14];
rz(-1.4461147365940448) q[14];
ry(1.570803844722973) q[15];
rz(2.8733348074919403) q[15];
ry(-3.798616266070809e-05) q[16];
rz(2.979251386553537) q[16];
ry(1.5707988041902277) q[17];
rz(-3.1413906120139856) q[17];
ry(3.0862250569187393) q[18];
rz(-1.590835773734459) q[18];
ry(3.103901038848311) q[19];
rz(1.5799029905092519) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.420826580988626) q[0];
rz(-1.546941801341167) q[0];
ry(3.0962741653835044) q[1];
rz(-0.05984123700947119) q[1];
ry(-0.42687049497427854) q[2];
rz(2.881774801628833) q[2];
ry(-0.7505593988398843) q[3];
rz(-0.6910396287173296) q[3];
ry(1.645128204656141) q[4];
rz(-1.5885514089791868) q[4];
ry(-1.5933130379982074) q[5];
rz(1.5254173914682247) q[5];
ry(-1.1932682719303356e-05) q[6];
rz(-0.296135648690706) q[6];
ry(0.00010614657090179946) q[7];
rz(1.6656104096165951) q[7];
ry(0.020576023566344668) q[8];
rz(0.004800940357697847) q[8];
ry(3.134322971778171) q[9];
rz(-1.2316201664341175) q[9];
ry(1.0066477062827879) q[10];
rz(0.8905469622478314) q[10];
ry(1.6836123659723976) q[11];
rz(2.2823656766388964) q[11];
ry(0.0013505251011823557) q[12];
rz(1.9370565033738556) q[12];
ry(1.5668224722862005) q[13];
rz(-3.1415584852247753) q[13];
ry(1.9298767799824645e-05) q[14];
rz(1.4461146545207846) q[14];
ry(-1.6912725140372231e-06) q[15];
rz(-2.873361832920706) q[15];
ry(3.1415408301910275) q[16];
rz(2.606652870499208) q[16];
ry(0.07971047780757257) q[17];
rz(0.18254179906936943) q[17];
ry(1.539757672008622) q[18];
rz(1.5725365551991588) q[18];
ry(0.022903042498521903) q[19];
rz(-1.577907124532513) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.186311340164947) q[0];
rz(-1.571970561325358) q[0];
ry(3.137861791640763) q[1];
rz(-0.059973543945556436) q[1];
ry(-0.3319681773625172) q[2];
rz(-2.95053218969052) q[2];
ry(3.0599314851701007) q[3];
rz(-2.4470015143154673) q[3];
ry(0.08412699669499513) q[4];
rz(1.57868647564088) q[4];
ry(3.133042196636909) q[5];
rz(0.032868892442730946) q[5];
ry(-1.570944860641397) q[6];
rz(1.2968548704990734) q[6];
ry(1.5701888026370865) q[7];
rz(-1.9030629404198434) q[7];
ry(1.5805027156093268) q[8];
rz(2.6789302159698516) q[8];
ry(-4.162165969804823e-05) q[9];
rz(-1.9132104180317793) q[9];
ry(3.1414002838663095) q[10];
rz(-1.5084082665141287) q[10];
ry(3.1415223990072105) q[11];
rz(2.289628283667018) q[11];
ry(-3.0183513615389437) q[12];
rz(-1.5681611674056564) q[12];
ry(-1.5742773537612953) q[13];
rz(-1.5714570233441405) q[13];
ry(-1.5707814422073945) q[14];
rz(0.5663363971575847) q[14];
ry(1.5709960921749655) q[15];
rz(0.33871196997137876) q[15];
ry(1.5708724303428356) q[16];
rz(1.5704923304266627) q[16];
ry(-3.141088682624611) q[17];
rz(2.225663482208897) q[17];
ry(-0.5367939859020234) q[18];
rz(-1.5729340120795259) q[18];
ry(-1.6736376714934686) q[19];
rz(-1.5868581605402494) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.9998612695784637) q[0];
rz(-3.140206541556672) q[0];
ry(-0.0035896502140584374) q[1];
rz(1.4496768549607633) q[1];
ry(1.4071027889718843) q[2];
rz(0.02652304714421803) q[2];
ry(1.5741949860980902) q[3];
rz(-1.9040265030687618) q[3];
ry(1.1137466570670056) q[4];
rz(-1.566359086505984) q[4];
ry(-0.11431218426948053) q[5];
rz(1.5220140400217934) q[5];
ry(-3.1415877602572984) q[6];
rz(1.2978117799209254) q[6];
ry(-3.1414500209956047) q[7];
rz(2.0797630623346746) q[7];
ry(-5.9216452012123e-05) q[8];
rz(2.0335745403032006) q[8];
ry(-2.618048263043754) q[9];
rz(2.7317896114194125) q[9];
ry(0.0018302235839220936) q[10];
rz(2.2689371503886084) q[10];
ry(-2.4708617189341) q[11];
rz(-1.5633031951901064) q[11];
ry(1.571672240911714) q[12];
rz(1.553170346304473) q[12];
ry(1.570854024908805) q[13];
rz(-1.567284841278248) q[13];
ry(-3.141578303373534) q[14];
rz(-2.5752528832725523) q[14];
ry(3.14159258601199) q[15];
rz(1.6983237736538017) q[15];
ry(-1.5708003274275635) q[16];
rz(1.6944583140212466) q[16];
ry(4.938986786679425e-05) q[17];
rz(1.8653082419578462) q[17];
ry(1.5707951904920838) q[18];
rz(-3.1415722510398822) q[18];
ry(-3.1415160507004374) q[19];
rz(-0.016530600134509573) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.570373471845759) q[0];
rz(2.9445137425020254) q[0];
ry(-3.1415790529146435) q[1];
rz(1.4620229626888381) q[1];
ry(-1.5827580616783377) q[2];
rz(1.4634854541851567) q[2];
ry(-3.1268574506496867) q[3];
rz(-2.7898361891737706) q[3];
ry(-2.890351893323076) q[4];
rz(-1.6830781550866487) q[4];
ry(1.5404074459451118) q[5];
rz(0.23663246097902757) q[5];
ry(-1.5706221483832385) q[6];
rz(-0.1470943496915727) q[6];
ry(0.0002467993227588394) q[7];
rz(-0.6072868167767957) q[7];
ry(1.57080418491344) q[8];
rz(-1.6569147257910997) q[8];
ry(3.1415814484324613) q[9];
rz(-0.17512882250568307) q[9];
ry(-3.1357197423936514) q[10];
rz(-1.8070353898406717) q[10];
ry(-1.5702906551649451) q[11];
rz(1.8049301211253583) q[11];
ry(-1.5729062679065917) q[12];
rz(3.0358986127434116) q[12];
ry(-1.4078812629915565) q[13];
rz(0.2327158155852489) q[13];
ry(-1.570766045697041) q[14];
rz(-1.671884463235413) q[14];
ry(-3.4606170959960995e-06) q[15];
rz(0.4446577392156357) q[15];
ry(3.141056670353247) q[16];
rz(1.5896229001089637) q[16];
ry(5.890655594241211e-06) q[17];
rz(-2.1046131946535054) q[17];
ry(-1.5707547991597481) q[18];
rz(3.038063040690927) q[18];
ry(-2.369886267223734) q[19];
rz(-2.9088324785388955) q[19];