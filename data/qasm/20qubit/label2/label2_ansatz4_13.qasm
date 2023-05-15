OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.025212295220129025) q[0];
rz(-1.4570353734783206) q[0];
ry(0.004674455794900346) q[1];
rz(-0.46478620334525367) q[1];
ry(-1.5890846149278905) q[2];
rz(-0.8214460762004229) q[2];
ry(1.5679061183319263) q[3];
rz(-2.721938198672209) q[3];
ry(0.0001105241482011096) q[4];
rz(1.9618412395078897) q[4];
ry(-3.141565875435652) q[5];
rz(-0.5180268977835122) q[5];
ry(-1.6054639699277464) q[6];
rz(2.103530585000744) q[6];
ry(1.570495568626571) q[7];
rz(1.2198633620926351) q[7];
ry(-0.0011624089156372364) q[8];
rz(-0.3880070963918695) q[8];
ry(-3.1408208961837163) q[9];
rz(-1.931158955332041) q[9];
ry(-1.5700358710913989) q[10];
rz(1.556797113814885) q[10];
ry(-1.5694105209201945) q[11];
rz(1.5490662294899928) q[11];
ry(-1.563957733110624) q[12];
rz(0.11526143598887728) q[12];
ry(1.575057102843675) q[13];
rz(1.292218472915562) q[13];
ry(3.1403278250272333) q[14];
rz(-0.3507552331931917) q[14];
ry(-3.1414486417542697) q[15];
rz(0.2695229623291097) q[15];
ry(-1.5734447858912273) q[16];
rz(-3.133200562783354) q[16];
ry(1.1583357710618567) q[17];
rz(-3.0545587847190476) q[17];
ry(-0.007354018696208854) q[18];
rz(-1.8860905598683655) q[18];
ry(-0.03546930046671193) q[19];
rz(-3.0135730421838747) q[19];
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
ry(-1.4638747273777386) q[0];
rz(-0.3547960583378549) q[0];
ry(2.1005586310840947) q[1];
rz(-0.6265534453407805) q[1];
ry(0.5403388347718105) q[2];
rz(-1.048584081051696) q[2];
ry(1.1870828235427904) q[3];
rz(1.250656631071847) q[3];
ry(0.1369219236950201) q[4];
rz(0.7353035700514596) q[4];
ry(1.60531089084997) q[5];
rz(-1.4070938805279682) q[5];
ry(-1.8937886264295098) q[6];
rz(-3.0394750594961963) q[6];
ry(1.0580576114045444) q[7];
rz(-1.3003337840036437) q[7];
ry(-2.9594282686526667) q[8];
rz(0.10711089979231936) q[8];
ry(-0.8459796456884793) q[9];
rz(1.1030802870049212) q[9];
ry(1.5678967942512239) q[10];
rz(-0.4638872082321687) q[10];
ry(1.5783084232903348) q[11];
rz(1.9483016686165315) q[11];
ry(-3.124413723433442) q[12];
rz(0.6507715704691845) q[12];
ry(0.010473129858563412) q[13];
rz(2.994019865699646) q[13];
ry(-0.09409789533960122) q[14];
rz(-1.4283064905941378) q[14];
ry(-3.141338464465038) q[15];
rz(-0.5391392606343288) q[15];
ry(0.9531812718276101) q[16];
rz(-0.11019023008813539) q[16];
ry(3.044259171225885) q[17];
rz(0.3911735239655991) q[17];
ry(-0.027157391122365482) q[18];
rz(-2.2359216049394246) q[18];
ry(3.105301264127433) q[19];
rz(-2.831053207107621) q[19];
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
ry(2.943623828995913) q[0];
rz(-0.3242835705973288) q[0];
ry(-1.6792697656878506) q[1];
rz(0.12867179177381782) q[1];
ry(0.8135181170684807) q[2];
rz(-0.0929870299353013) q[2];
ry(-2.333664150227246) q[3];
rz(-0.9992107126016824) q[3];
ry(-2.963142548877611) q[4];
rz(-1.9454939179126125) q[4];
ry(-0.19508083300274937) q[5];
rz(-0.1685858866792123) q[5];
ry(-0.0011220148562420817) q[6];
rz(1.336281831485579) q[6];
ry(-3.139228978669565) q[7];
rz(1.532699101981911) q[7];
ry(2.168244191565993) q[8];
rz(0.2008742896372988) q[8];
ry(-2.3851075562647917) q[9];
rz(0.7176481569741839) q[9];
ry(0.0009130314706684574) q[10];
rz(1.7449636785944644) q[10];
ry(-3.1241352597110974) q[11];
rz(0.08225504508159817) q[11];
ry(-3.1376901119905534) q[12];
rz(-2.6423086137834018) q[12];
ry(0.005690829175865138) q[13];
rz(1.8599893415492548) q[13];
ry(-0.0021757266954306495) q[14];
rz(1.29062027567495) q[14];
ry(-3.1203174470771793) q[15];
rz(0.9765803378826793) q[15];
ry(-2.5686185537542987) q[16];
rz(2.735748395178571) q[16];
ry(-0.7393851159691665) q[17];
rz(-0.7372460934045487) q[17];
ry(1.158510463333653) q[18];
rz(-2.3484696191237653) q[18];
ry(-1.4374037706684213) q[19];
rz(-0.3816250034065692) q[19];
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
ry(-3.0068040286326885) q[0];
rz(-2.6220837586305046) q[0];
ry(-0.1333875547992811) q[1];
rz(-2.962082885618758) q[1];
ry(2.6139527605989095) q[2];
rz(3.0188021630064563) q[2];
ry(0.812607839639248) q[3];
rz(0.846279247086104) q[3];
ry(1.6795043824574911) q[4];
rz(0.03629540864499391) q[4];
ry(-1.1893821155108935) q[5];
rz(1.5685079563944475) q[5];
ry(-2.4891055182149944) q[6];
rz(2.660643995644833) q[6];
ry(2.4980898301290018) q[7];
rz(-0.5676891838261771) q[7];
ry(-1.7672084585975072) q[8];
rz(-2.951969317023232) q[8];
ry(-0.6587700774797813) q[9];
rz(-2.3193194131032424) q[9];
ry(-1.5483410754430067) q[10];
rz(0.3415079534141532) q[10];
ry(1.5579409726614184) q[11];
rz(3.1398137051163424) q[11];
ry(-0.07090496454031268) q[12];
rz(0.0630366365484436) q[12];
ry(3.0609717236032368) q[13];
rz(3.0129237901411896) q[13];
ry(-0.06222007202539892) q[14];
rz(0.5581456771193113) q[14];
ry(-3.1355620632523173) q[15];
rz(2.343767211942084) q[15];
ry(0.0036230116783139968) q[16];
rz(-3.0889426351318128) q[16];
ry(3.066920386775764) q[17];
rz(1.3595049923408729) q[17];
ry(-0.8910597581038571) q[18];
rz(-0.18953407334154537) q[18];
ry(1.125727124740971) q[19];
rz(2.6142860865931787) q[19];
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
ry(3.1133166308191815) q[0];
rz(0.5847833466662973) q[0];
ry(3.1278947199153135) q[1];
rz(1.1054385253331915) q[1];
ry(-1.578167118584938) q[2];
rz(1.643953436853614) q[2];
ry(1.5673956290288942) q[3];
rz(1.4414929972315558) q[3];
ry(1.6718718086301678) q[4];
rz(0.5339840241820605) q[4];
ry(1.6890526398455834) q[5];
rz(-1.5931919446327996) q[5];
ry(-1.0156142766954064) q[6];
rz(-1.8122372948872405) q[6];
ry(2.1503999343428752) q[7];
rz(-1.4166316484658719) q[7];
ry(-0.01809444661876813) q[8];
rz(-2.286433528926447) q[8];
ry(-0.05051166157962151) q[9];
rz(-2.65005658408857) q[9];
ry(0.001048241995425841) q[10];
rz(2.8140868804220074) q[10];
ry(1.8820611925365875) q[11];
rz(-0.022502533310175817) q[11];
ry(1.5771061165622822) q[12];
rz(-0.10201764041122097) q[12];
ry(-1.5689194573628973) q[13];
rz(0.19964554166508464) q[13];
ry(-0.029613688187534657) q[14];
rz(0.0944647614754901) q[14];
ry(0.01733492409929054) q[15];
rz(2.4081268981180797) q[15];
ry(-3.0463529438017787) q[16];
rz(1.0333015036864897) q[16];
ry(-0.37248416146553875) q[17];
rz(-0.7772884633370438) q[17];
ry(2.2913468895314764) q[18];
rz(2.165561817285355) q[18];
ry(-2.3230224447363232) q[19];
rz(-2.4033506371501643) q[19];
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
ry(-2.0556092190328137) q[0];
rz(-1.4491526120346612) q[0];
ry(-0.0917928839294525) q[1];
rz(-1.9440641002517376) q[1];
ry(1.4911968579010573) q[2];
rz(0.5290048061107453) q[2];
ry(0.7819459867626204) q[3];
rz(-0.2929234123094995) q[3];
ry(2.8792029592493877) q[4];
rz(2.2342223526401317) q[4];
ry(-1.1263583636117611) q[5];
rz(-1.2502876953117337) q[5];
ry(0.8594685356465483) q[6];
rz(0.1599202184633013) q[6];
ry(-2.362939000812575) q[7];
rz(-2.4402008020985324) q[7];
ry(0.3519524170845343) q[8];
rz(-2.7190144373473246) q[8];
ry(-2.5518521249643045) q[9];
rz(-1.03347104638745) q[9];
ry(1.5486515192858592) q[10];
rz(0.44205744257425367) q[10];
ry(1.5831117775557335) q[11];
rz(2.430087884645158) q[11];
ry(1.5820451613647935) q[12];
rz(-3.1097600477317813) q[12];
ry(-1.5493783069861653) q[13];
rz(-0.05645846642776142) q[13];
ry(-0.0009958171552669497) q[14];
rz(-2.8779528523954836) q[14];
ry(-3.140843846157441) q[15];
rz(-1.8628886687617707) q[15];
ry(-0.027095412956481067) q[16];
rz(-2.8286262435494613) q[16];
ry(-0.05226083261654805) q[17];
rz(0.46766149582054256) q[17];
ry(1.6134216595454365) q[18];
rz(-2.236667407620084) q[18];
ry(-1.8773690346594267) q[19];
rz(-1.7873396931123349) q[19];
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
ry(-0.03280404200877563) q[0];
rz(0.8322721996793083) q[0];
ry(1.1090379208335275) q[1];
rz(2.6265834249155504) q[1];
ry(0.04408114778657968) q[2];
rz(3.134023264192761) q[2];
ry(-0.039492566563950095) q[3];
rz(-0.6388559255695447) q[3];
ry(-0.005176261511223202) q[4];
rz(1.0700092141899586) q[4];
ry(3.1182079745854767) q[5];
rz(1.4072820475355918) q[5];
ry(-0.03853904988133016) q[6];
rz(-0.656333788428614) q[6];
ry(-3.0543308597994945) q[7];
rz(1.2147283188408977) q[7];
ry(1.683965048152514) q[8];
rz(1.2895762414608452) q[8];
ry(-0.8988587747345897) q[9];
rz(-2.873976980489865) q[9];
ry(-0.16785717859412674) q[10];
rz(1.8198456106694634) q[10];
ry(-2.9934321232756163) q[11];
rz(-0.23979627166968323) q[11];
ry(0.3879944410746159) q[12];
rz(-1.7165033253550854) q[12];
ry(-1.21819127688137) q[13];
rz(2.2901775462996197) q[13];
ry(2.4481953588106866) q[14];
rz(2.6113132152973138) q[14];
ry(-0.41790927101815234) q[15];
rz(-0.4247117558620562) q[15];
ry(-3.1089484711554576) q[16];
rz(0.5556817023620642) q[16];
ry(-0.7927702980931005) q[17];
rz(-2.108493565848623) q[17];
ry(1.5580236037138357) q[18];
rz(2.5254667820096195) q[18];
ry(-0.549525205415513) q[19];
rz(-2.0834278475497774) q[19];
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
ry(2.036200265590926) q[0];
rz(-1.534246095287866) q[0];
ry(2.143541620145215) q[1];
rz(-2.172446539082472) q[1];
ry(-1.0610548587451332) q[2];
rz(-1.482403080318619) q[2];
ry(0.633136763779194) q[3];
rz(2.2398482449104344) q[3];
ry(2.9357006789700484) q[4];
rz(-2.9586147925192225) q[4];
ry(0.74342847288224) q[5];
rz(0.9765798017150723) q[5];
ry(1.39232442437153) q[6];
rz(0.4040991509844156) q[6];
ry(1.3644617381092616) q[7];
rz(-2.605617755684411) q[7];
ry(0.05163681276085984) q[8];
rz(0.9400120348202279) q[8];
ry(0.23333859586790062) q[9];
rz(-1.8018895534044646) q[9];
ry(-0.04852184834652106) q[10];
rz(-2.8266294862121817) q[10];
ry(-0.06719381413686953) q[11];
rz(-2.5535150468579064) q[11];
ry(0.003815765522092297) q[12];
rz(3.084269646829264) q[12];
ry(-0.0026138337068308815) q[13];
rz(2.2715115096013907) q[13];
ry(0.9482007656409823) q[14];
rz(-0.37003805754553587) q[14];
ry(-1.8772263899419368) q[15];
rz(-1.8613921031761538) q[15];
ry(0.017499271942588483) q[16];
rz(0.2639542113711176) q[16];
ry(-0.0093388729722097) q[17];
rz(1.3671167870597039) q[17];
ry(3.057460092589589) q[18];
rz(2.6307267249425412) q[18];
ry(1.39024806709885) q[19];
rz(1.4180986511367137) q[19];
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
ry(3.050884206345691) q[0];
rz(1.114738330838437) q[0];
ry(0.08572915351242728) q[1];
rz(-1.7863434465245407) q[1];
ry(-0.004477162853129175) q[2];
rz(2.227882586383605) q[2];
ry(3.1385735954900307) q[3];
rz(-3.0066250231841316) q[3];
ry(-3.1399842879114623) q[4];
rz(1.2710780256544432) q[4];
ry(-3.133632691406338) q[5];
rz(-0.31895052877385793) q[5];
ry(-3.113922296543987) q[6];
rz(-1.5168705399009765) q[6];
ry(-3.1320975167780665) q[7];
rz(2.3722985957497755) q[7];
ry(2.130997565776997) q[8];
rz(-2.7034935592360503) q[8];
ry(1.529317926684543) q[9];
rz(-1.4908409533819738) q[9];
ry(1.9821711765604328) q[10];
rz(2.9508576375285407) q[10];
ry(-1.1172306539992487) q[11];
rz(0.03680385855028042) q[11];
ry(2.094033558824684) q[12];
rz(-0.8650459563759014) q[12];
ry(-2.1102547335763657) q[13];
rz(2.9574122354314114) q[13];
ry(0.6089104247668552) q[14];
rz(-2.6132005615391165) q[14];
ry(-2.393327481413982) q[15];
rz(-1.9778482383343752) q[15];
ry(2.276955812062252) q[16];
rz(-0.841079387290774) q[16];
ry(-2.3084741493405225) q[17];
rz(-2.9012252641658263) q[17];
ry(2.375649170599543) q[18];
rz(-3.074056673081594) q[18];
ry(-1.0438605831761725) q[19];
rz(-2.7384987799219993) q[19];
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
ry(-2.531247274147149) q[0];
rz(-0.7016781381170762) q[0];
ry(-0.24202854394210488) q[1];
rz(2.337631447442913) q[1];
ry(-2.3336842117979457) q[2];
rz(-2.51664042560216) q[2];
ry(-2.326772691652887) q[3];
rz(-1.1443024511146742) q[3];
ry(-3.0855119005033584) q[4];
rz(-1.4024214622287392) q[4];
ry(-2.7528858847872013) q[5];
rz(-1.6943387423658762) q[5];
ry(1.1705435228810455) q[6];
rz(-0.6378614619751826) q[6];
ry(1.975945847399501) q[7];
rz(-0.7138990862740808) q[7];
ry(-2.747108556670586) q[8];
rz(-2.046776841815592) q[8];
ry(-0.2737369241798504) q[9];
rz(-2.440621439538393) q[9];
ry(-2.1450279339292893) q[10];
rz(1.1716230325618593) q[10];
ry(1.0030447179235846) q[11];
rz(1.2855543598025818) q[11];
ry(-3.1069662098382778) q[12];
rz(0.5993045952558997) q[12];
ry(0.019777700911099566) q[13];
rz(-1.2516440507515454) q[13];
ry(1.8051289241959785) q[14];
rz(-2.27228160196467) q[14];
ry(-1.9863945436760095) q[15];
rz(-0.2158201126939844) q[15];
ry(-0.0023941958908677563) q[16];
rz(0.045798792740966654) q[16];
ry(-3.1383679437183094) q[17];
rz(0.75485548301029) q[17];
ry(-2.8238009191788227) q[18];
rz(-0.7013599148120662) q[18];
ry(-1.6695657641959283) q[19];
rz(1.5638731040246165) q[19];
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
ry(-1.2321064327181723) q[0];
rz(2.961847270243478) q[0];
ry(-2.2824585414935266) q[1];
rz(-2.715507156453263) q[1];
ry(0.008186202251757813) q[2];
rz(2.299183022372091) q[2];
ry(0.021940446314585138) q[3];
rz(-1.3495146354760479) q[3];
ry(-0.006393615041755338) q[4];
rz(2.117273338676884) q[4];
ry(-3.1303290294084607) q[5];
rz(2.470392188596027) q[5];
ry(-1.8302436254847443) q[6];
rz(-2.485147784048741) q[6];
ry(-1.2840626335877587) q[7];
rz(0.8455784683295908) q[7];
ry(0.0245142112302128) q[8];
rz(0.3481104296144903) q[8];
ry(3.0341685717211218) q[9];
rz(0.1100031719075944) q[9];
ry(-2.4698190258674715) q[10];
rz(-1.7621136983764591) q[10];
ry(0.6747442238213974) q[11];
rz(-2.0802308695237546) q[11];
ry(0.1254429990044157) q[12];
rz(-1.130327124365074) q[12];
ry(2.9601015678240783) q[13];
rz(-0.3362669746591392) q[13];
ry(2.255110923487794) q[14];
rz(1.2599908328426268) q[14];
ry(-0.43638986865602636) q[15];
rz(1.1302273131000067) q[15];
ry(-3.1399922140933336) q[16];
rz(1.043290998151167) q[16];
ry(3.1381377774100576) q[17];
rz(-2.733924141944197) q[17];
ry(-0.03228744156637742) q[18];
rz(2.5454111043970884) q[18];
ry(-1.8536361532285737) q[19];
rz(2.8453931811215205) q[19];
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
ry(0.059644186640883744) q[0];
rz(0.4522812618853739) q[0];
ry(1.465155418147943) q[1];
rz(-1.6304722630019866) q[1];
ry(-1.608569268692678) q[2];
rz(-3.129107473473789) q[2];
ry(-1.5353388159204986) q[3];
rz(3.1128567265710623) q[3];
ry(1.7739597958309417) q[4];
rz(0.18700527008659806) q[4];
ry(1.999785079121021) q[5];
rz(0.9877716887790763) q[5];
ry(-2.802531979935909) q[6];
rz(-2.508355715042583) q[6];
ry(3.1319056186021936) q[7];
rz(0.7862869715504713) q[7];
ry(0.16739200970288945) q[8];
rz(-1.284987601833806) q[8];
ry(0.1662832479090198) q[9];
rz(-0.019085796627051305) q[9];
ry(-1.4661051112805366) q[10];
rz(1.6701350379011486) q[10];
ry(2.4584539418658817) q[11];
rz(3.121075185844415) q[11];
ry(-0.043441121861406806) q[12];
rz(0.3972575791560082) q[12];
ry(-3.1097640954504806) q[13];
rz(1.2812180776445494) q[13];
ry(-1.5300879188254362) q[14];
rz(-2.1010430331550976) q[14];
ry(1.7057883481218497) q[15];
rz(2.4517841690292212) q[15];
ry(-0.022492955290431382) q[16];
rz(-1.3008283830387484) q[16];
ry(-0.039184506362681754) q[17];
rz(-3.137738311345362) q[17];
ry(-1.3833978277768118) q[18];
rz(-1.1422324245206514) q[18];
ry(2.3246313857558114) q[19];
rz(-1.721766848384176) q[19];
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
ry(-2.1740826446864823) q[0];
rz(1.5559014090028995) q[0];
ry(-1.447111137339266) q[1];
rz(1.0738402347980465) q[1];
ry(3.003185246027998) q[2];
rz(-0.17285646422512205) q[2];
ry(-0.14191932885578762) q[3];
rz(2.6532307923219465) q[3];
ry(3.1168907062696176) q[4];
rz(-3.0691666391526327) q[4];
ry(-0.009607482800744208) q[5];
rz(2.143676683157547) q[5];
ry(-0.06282766029126248) q[6];
rz(-2.138520084100091) q[6];
ry(0.013597209611337922) q[7];
rz(0.25184361971007385) q[7];
ry(-0.022007769807681044) q[8];
rz(-2.610040428885022) q[8];
ry(3.1209275768899305) q[9];
rz(0.5500952345037523) q[9];
ry(1.1559845878292723) q[10];
rz(1.687200461979213) q[10];
ry(-1.347309566858345) q[11];
rz(-1.0824626723217547) q[11];
ry(0.03667207170539943) q[12];
rz(-1.1001870486286753) q[12];
ry(-1.4782941781727643) q[13];
rz(0.1392680286241905) q[13];
ry(0.46416106574946414) q[14];
rz(-2.9405568700993987) q[14];
ry(-2.8884140683274544) q[15];
rz(-2.5265149161752674) q[15];
ry(-3.13566309787529) q[16];
rz(-1.620574584025217) q[16];
ry(0.0474583822447725) q[17];
rz(-0.8467778602034776) q[17];
ry(-2.7823859319199467) q[18];
rz(-0.3795822463454491) q[18];
ry(-1.8788705611144518) q[19];
rz(0.1189538576188669) q[19];
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
ry(2.309884521585235) q[0];
rz(-0.42689495875198435) q[0];
ry(-2.0853683969295433) q[1];
rz(0.2803317739818737) q[1];
ry(-0.08145303004326358) q[2];
rz(1.482022426884292) q[2];
ry(-3.0811905331065987) q[3];
rz(1.3598400500051944) q[3];
ry(-1.782751870788871) q[4];
rz(-1.341927803061873) q[4];
ry(1.590000319621149) q[5];
rz(-3.07587694836811) q[5];
ry(2.1609560952888485) q[6];
rz(0.40446914337888007) q[6];
ry(-1.4198544015299648) q[7];
rz(-2.599235702161512) q[7];
ry(3.1204554030411953) q[8];
rz(-2.5411518939644457) q[8];
ry(-0.0031977469405992665) q[9];
rz(-1.041125372681937) q[9];
ry(3.104108582777032) q[10];
rz(0.7316046563068497) q[10];
ry(-0.05070622428349333) q[11];
rz(-2.314123343058416) q[11];
ry(-3.1041947527351157) q[12];
rz(1.2326097752001983) q[12];
ry(3.0944830677189143) q[13];
rz(0.13305461858309942) q[13];
ry(3.139495256751165) q[14];
rz(0.7529435784484627) q[14];
ry(3.138217285394781) q[15];
rz(2.5840350987830414) q[15];
ry(-0.3555927787520634) q[16];
rz(-2.3631065924342876) q[16];
ry(-2.814847434097545) q[17];
rz(-0.9585999379694269) q[17];
ry(-1.7717454007533977) q[18];
rz(-2.609983315173616) q[18];
ry(1.224516816068833) q[19];
rz(2.5500626059433507) q[19];
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
ry(2.707305329926773) q[0];
rz(-0.7592823968826599) q[0];
ry(0.9379127328416001) q[1];
rz(-2.5247919048408063) q[1];
ry(-1.5413643489143452) q[2];
rz(0.008764922536605544) q[2];
ry(-1.6214592760398625) q[3];
rz(0.024343486730687678) q[3];
ry(-3.096936262062619) q[4];
rz(-2.4653764958785342) q[4];
ry(-3.1395691010256215) q[5];
rz(1.2322530408241334) q[5];
ry(0.2074395168157067) q[6];
rz(-0.9401655640800508) q[6];
ry(3.0247124446295164) q[7];
rz(0.03149518469984368) q[7];
ry(-0.0019657823781448077) q[8];
rz(1.1606493106214861) q[8];
ry(-3.1399651217195363) q[9];
rz(-0.17756193087590813) q[9];
ry(-2.3644855438364565) q[10];
rz(-2.631409317785487) q[10];
ry(2.255143189244457) q[11];
rz(-2.5593600176023066) q[11];
ry(-3.103542376520263) q[12];
rz(1.5029602641524278) q[12];
ry(1.5181729132693933) q[13];
rz(-3.102130038524685) q[13];
ry(-1.3623237775120536) q[14];
rz(0.8110035553441821) q[14];
ry(-1.7684336487568943) q[15];
rz(2.853295085805686) q[15];
ry(2.855652443470184) q[16];
rz(1.878440687246908) q[16];
ry(-1.6291913457274243) q[17];
rz(0.596249462605786) q[17];
ry(-1.644256103046196) q[18];
rz(-2.1301158370696998) q[18];
ry(-2.563581773064757) q[19];
rz(0.7307295997168081) q[19];
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
ry(-2.620829620970526) q[0];
rz(-1.9799809315737904) q[0];
ry(-2.3053463343954665) q[1];
rz(-2.5824278861040737) q[1];
ry(1.2161294974934727) q[2];
rz(2.5030532253605116) q[2];
ry(1.9245173293162214) q[3];
rz(0.6379023452461389) q[3];
ry(3.092916472215959) q[4];
rz(-1.0294497405670018) q[4];
ry(3.102588762268093) q[5];
rz(-2.976575504200518) q[5];
ry(-1.6447563301849208) q[6];
rz(0.5630935731054221) q[6];
ry(-1.6187052001444335) q[7];
rz(2.0851455360174906) q[7];
ry(0.002188914762694115) q[8];
rz(-1.3906112516593956) q[8];
ry(0.021052090480342168) q[9];
rz(0.3020533732655375) q[9];
ry(-1.2441470743523089) q[10];
rz(0.48139318636553785) q[10];
ry(-0.9610674694114358) q[11];
rz(-2.9788973472274543) q[11];
ry(-2.9829330249962958) q[12];
rz(-0.8654460663243401) q[12];
ry(2.991285848835476) q[13];
rz(2.2296758366299674) q[13];
ry(0.015693802648721614) q[14];
rz(1.3653139450320124) q[14];
ry(3.1236424294449847) q[15];
rz(-0.6571548605368817) q[15];
ry(3.1396638028649084) q[16];
rz(0.5084218525059024) q[16];
ry(-3.139397631351545) q[17];
rz(2.904875804229514) q[17];
ry(-0.009972852941505295) q[18];
rz(0.9403945750813056) q[18];
ry(3.1382224356254547) q[19];
rz(-2.9238614371404705) q[19];
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
ry(0.7684316515751921) q[0];
rz(-2.7525875300011675) q[0];
ry(2.1930954081367786) q[1];
rz(0.7994643196495339) q[1];
ry(0.6250520446532786) q[2];
rz(-1.0128139694320863) q[2];
ry(2.5355551591728203) q[3];
rz(2.117793753392487) q[3];
ry(-1.677723350864606) q[4];
rz(1.322773003527173) q[4];
ry(0.31217213072571853) q[5];
rz(-2.426684046998491) q[5];
ry(-3.0521354588539706) q[6];
rz(-1.5251702216120153) q[6];
ry(0.09200580450568374) q[7];
rz(0.6736096397425198) q[7];
ry(0.05785782065071032) q[8];
rz(0.9677538819451912) q[8];
ry(3.098541839223313) q[9];
rz(-1.8884107369806695) q[9];
ry(-0.4773551231236235) q[10];
rz(-0.1139978381671467) q[10];
ry(-0.5544035833083543) q[11];
rz(0.35838026169007087) q[11];
ry(2.7908639195673057) q[12];
rz(2.9156198885410807) q[12];
ry(2.803089174132089) q[13];
rz(2.9129947683618354) q[13];
ry(0.17932071223962165) q[14];
rz(1.679836575371982) q[14];
ry(0.10482163244908714) q[15];
rz(1.0432092841890372) q[15];
ry(-1.3174139911088885) q[16];
rz(-2.5834745637535597) q[16];
ry(-0.3801343681368239) q[17];
rz(1.7273966484135552) q[17];
ry(0.9781285031393746) q[18];
rz(0.8810981286802313) q[18];
ry(1.7223462079015508) q[19];
rz(2.239320221033709) q[19];