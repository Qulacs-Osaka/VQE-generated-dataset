OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.0949984019539074) q[0];
rz(-1.831375338972638) q[0];
ry(1.2322910850602131) q[1];
rz(-1.722869897885309) q[1];
ry(-2.6158849896434275) q[2];
rz(-2.3137104962464528) q[2];
ry(-0.5035199680754312) q[3];
rz(-0.7043688651215758) q[3];
ry(3.108750263589159) q[4];
rz(1.4897715081102456) q[4];
ry(-1.5302229880559277) q[5];
rz(-1.9471704109528778) q[5];
ry(-2.2995760731147463) q[6];
rz(-0.7667440645415518) q[6];
ry(-1.5321089556565717) q[7];
rz(0.4319529874856728) q[7];
ry(0.009256444558943144) q[8];
rz(2.3021675150693097) q[8];
ry(1.5702468803381102) q[9];
rz(0.29632026754985485) q[9];
ry(1.5730255263142334) q[10];
rz(0.598031139301731) q[10];
ry(-1.396272999140554) q[11];
rz(-3.141022023930156) q[11];
ry(3.138448236246438) q[12];
rz(-1.319298102757963) q[12];
ry(1.9065800913750799) q[13];
rz(-0.009213490983726447) q[13];
ry(1.748751871260077) q[14];
rz(0.0746143934430239) q[14];
ry(1.2660572404360266) q[15];
rz(1.3745823174697347) q[15];
ry(3.133212201597901) q[16];
rz(1.6908367887794646) q[16];
ry(0.79836650185583) q[17];
rz(-0.09651822091316031) q[17];
ry(0.16244486045996975) q[18];
rz(-2.107800549333346) q[18];
ry(0.5802817050469765) q[19];
rz(1.5199783324651528) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.42804413963760485) q[0];
rz(1.56833131453123) q[0];
ry(0.009788132498178115) q[1];
rz(1.438157440330853) q[1];
ry(3.139186017694239) q[2];
rz(-2.308048130178035) q[2];
ry(1.1596461546870476) q[3];
rz(-0.4885730617642419) q[3];
ry(-3.0122019931313715) q[4];
rz(1.040146408669502) q[4];
ry(2.0525570105950512) q[5];
rz(-1.7155105211532213) q[5];
ry(1.6247737463503578) q[6];
rz(2.8948256084503816) q[6];
ry(-1.4568715565366874) q[7];
rz(0.3169443449767146) q[7];
ry(-1.5704249631063023) q[8];
rz(-2.6297278824733374) q[8];
ry(-2.1412031464999988) q[9];
rz(-2.8107995197186004) q[9];
ry(2.5916410653648088) q[10];
rz(2.8492940148913912) q[10];
ry(1.5724586366711053) q[11];
rz(-2.69220315338306) q[11];
ry(0.0017245261069671614) q[12];
rz(-1.7120951819191377) q[12];
ry(-1.5789020077946583) q[13];
rz(-3.0957806443725486) q[13];
ry(2.530636966875377) q[14];
rz(0.42934518353240436) q[14];
ry(1.0823658286361875) q[15];
rz(-1.8534925914555525) q[15];
ry(-0.7595647723073256) q[16];
rz(2.8926844689130973) q[16];
ry(2.388388292005812) q[17];
rz(-3.0234245965320152) q[17];
ry(-2.718678940875865) q[18];
rz(0.046795302548849345) q[18];
ry(-1.32472621349209) q[19];
rz(-0.7340538081400636) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.438627173554322) q[0];
rz(-1.8741655994244368) q[0];
ry(-1.5101768228464163) q[1];
rz(-0.9436438514926553) q[1];
ry(3.0454396911989234) q[2];
rz(-0.7357651686024046) q[2];
ry(-0.30839108799882897) q[3];
rz(1.6061331220168016) q[3];
ry(-0.782790869984864) q[4];
rz(0.6299473293506361) q[4];
ry(1.603394913131348) q[5];
rz(-2.079361585980618) q[5];
ry(-1.6707638465939247) q[6];
rz(-0.00976324677030542) q[6];
ry(1.5693132378700003) q[7];
rz(-0.42475858898201496) q[7];
ry(-0.3862171655889437) q[8];
rz(-0.09376218735015751) q[8];
ry(1.336728891047083) q[9];
rz(-1.6401792092957799) q[9];
ry(-1.0993639242249351) q[10];
rz(-0.6921285272539256) q[10];
ry(-1.6624457520942288) q[11];
rz(-1.8918465227038705) q[11];
ry(-1.5697780059148236) q[12];
rz(-0.35063749596967414) q[12];
ry(0.5277044535676305) q[13];
rz(-3.1107840781505485) q[13];
ry(-0.030382951914413328) q[14];
rz(-0.44283445024136725) q[14];
ry(-0.006729612048856026) q[15];
rz(-1.455010055464453) q[15];
ry(3.141394100747762) q[16];
rz(1.7910078973974937) q[16];
ry(2.418490043427521) q[17];
rz(1.3072366746765205) q[17];
ry(-0.14908208013164878) q[18];
rz(-2.312550575187963) q[18];
ry(-2.7289773355604185) q[19];
rz(2.3039555967070116) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.2249296496721604) q[0];
rz(-0.057864082992764625) q[0];
ry(-3.1008842498749734) q[1];
rz(1.881603160380899) q[1];
ry(-3.120224192897937) q[2];
rz(-0.7699968099049118) q[2];
ry(-0.15097622528384663) q[3];
rz(1.3666254595860972) q[3];
ry(-0.28548452343275293) q[4];
rz(-0.1710829172441528) q[4];
ry(-0.24872669287639987) q[5];
rz(1.4237766243695582) q[5];
ry(-1.5604312312435762) q[6];
rz(-0.17542310621127297) q[6];
ry(-2.7199000346319875) q[7];
rz(-1.3338916200934072) q[7];
ry(-1.1542227469618498) q[8];
rz(-1.2458779530920259) q[8];
ry(1.1849270624480033) q[9];
rz(-0.6161637515073721) q[9];
ry(-0.5621478268836553) q[10];
rz(-1.645448611270028) q[10];
ry(-2.7305013474325666) q[11];
rz(-1.9440561476228677) q[11];
ry(0.046187521611324514) q[12];
rz(-1.7384796585092774) q[12];
ry(-1.5639663531373733) q[13];
rz(-0.4571282886142665) q[13];
ry(-2.1566748800778983) q[14];
rz(0.051778693344806115) q[14];
ry(2.0752644055418465) q[15];
rz(2.8619432966272926) q[15];
ry(0.05111040799426709) q[16];
rz(-2.2070653884813582) q[16];
ry(-0.8436081637437242) q[17];
rz(-1.3979132179628049) q[17];
ry(2.6871980423214485) q[18];
rz(2.39031297705647) q[18];
ry(-2.062997341250896) q[19];
rz(-0.8505927851057038) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.1984199867885295) q[0];
rz(-0.9349305847066037) q[0];
ry(2.976848352767439) q[1];
rz(0.17083749718205787) q[1];
ry(0.4164363387120852) q[2];
rz(1.5176260072763865) q[2];
ry(0.1208413516214053) q[3];
rz(0.21285684420561868) q[3];
ry(0.683047359507661) q[4];
rz(-0.3635275956342904) q[4];
ry(-1.718411037790177) q[5];
rz(0.9815240467489618) q[5];
ry(3.1294550262476117) q[6];
rz(-0.32678040460189256) q[6];
ry(3.1183280870422787) q[7];
rz(2.3636502730315008) q[7];
ry(3.099616733769788) q[8];
rz(-1.234487022803285) q[8];
ry(-0.8725309043236171) q[9];
rz(0.3260922702142569) q[9];
ry(1.6575458532741443) q[10];
rz(1.5659001853025627) q[10];
ry(-1.9720292182901655) q[11];
rz(-2.4141768670479227) q[11];
ry(0.27299805234305463) q[12];
rz(2.8215429825142384) q[12];
ry(0.35098432914754785) q[13];
rz(-2.505887105517136) q[13];
ry(1.586559369583016) q[14];
rz(-0.6336805506018012) q[14];
ry(0.02181493619422405) q[15];
rz(-0.8988667213925279) q[15];
ry(3.1387280401539406) q[16];
rz(-3.0907451057191153) q[16];
ry(-0.5158597794165827) q[17];
rz(0.4852856443380764) q[17];
ry(-0.0501274643600083) q[18];
rz(2.3301203523062424) q[18];
ry(-1.2875625477228385) q[19];
rz(1.357964231938765) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.3201233092823137) q[0];
rz(1.1915113560417292) q[0];
ry(-3.1300802921433957) q[1];
rz(-1.4154006222895512) q[1];
ry(0.03068707397937498) q[2];
rz(-1.066650676677929) q[2];
ry(-1.669765644531048) q[3];
rz(-0.4575999653630722) q[3];
ry(3.095043146003761) q[4];
rz(2.8304656969492274) q[4];
ry(1.5388339355784355) q[5];
rz(1.5844182600292405) q[5];
ry(3.09970637655852) q[6];
rz(-1.9043458143088083) q[6];
ry(0.17958271214748422) q[7];
rz(0.241461808203507) q[7];
ry(-1.1790581175808699) q[8];
rz(1.0551806300196045) q[8];
ry(1.893774851949508) q[9];
rz(-1.0974687421462628) q[9];
ry(1.145993061492086) q[10];
rz(0.19655387484727577) q[10];
ry(-2.755058266108706) q[11];
rz(-0.1773628850020419) q[11];
ry(-1.182054052705108) q[12];
rz(-2.5723083654187016) q[12];
ry(2.7550234450967794) q[13];
rz(3.0833625579832282) q[13];
ry(0.18794488809038667) q[14];
rz(1.8365641880294232) q[14];
ry(0.3428085755360133) q[15];
rz(-0.8195815601957417) q[15];
ry(-0.7761622657344011) q[16];
rz(-1.9437868083160659) q[16];
ry(2.556507615140553) q[17];
rz(-1.8529981059228589) q[17];
ry(1.4202114209348817) q[18];
rz(-2.477733046887449) q[18];
ry(-2.0672523954779574) q[19];
rz(0.67153063478832) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.9563971677604592) q[0];
rz(-2.140035719054331) q[0];
ry(-2.5983794447661053) q[1];
rz(-1.0198088658238191) q[1];
ry(-0.641191315012576) q[2];
rz(-0.8009772630370282) q[2];
ry(3.053580686398371) q[3];
rz(-2.6900957521883067) q[3];
ry(-0.030944061544293536) q[4];
rz(2.9749159064251636) q[4];
ry(1.7954130492910743) q[5];
rz(1.8394131247759127) q[5];
ry(-2.6581725750834693) q[6];
rz(-0.09540599660742577) q[6];
ry(2.6177999468305493) q[7];
rz(1.0298169039562863) q[7];
ry(2.3791318483381256) q[8];
rz(1.5382435830865202) q[8];
ry(1.250827080962918) q[9];
rz(-2.9856463524128825) q[9];
ry(-0.632619781821027) q[10];
rz(0.43374398148807614) q[10];
ry(-3.0943834426993178) q[11];
rz(-1.0653594198383298) q[11];
ry(2.2902806114217746) q[12];
rz(-2.2035449127231868) q[12];
ry(-2.3760645705141097) q[13];
rz(0.1135416875190069) q[13];
ry(0.035380305493335665) q[14];
rz(1.6252494562351212) q[14];
ry(-0.11269012917247986) q[15];
rz(-3.0348050264506456) q[15];
ry(3.086183950098363) q[16];
rz(2.1180559519505824) q[16];
ry(-0.9321799564019093) q[17];
rz(-1.5529126715329304) q[17];
ry(-1.6219878766393798) q[18];
rz(2.1490409294033777) q[18];
ry(1.3313802930788317) q[19];
rz(1.315281712122086) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.1309645613156585) q[0];
rz(-0.5318896290024435) q[0];
ry(1.3401692146700643) q[1];
rz(0.3390200042858714) q[1];
ry(-3.1238923797986113) q[2];
rz(-0.5027292651442972) q[2];
ry(-3.0876857100011024) q[3];
rz(0.8448784686124524) q[3];
ry(-1.578823832634583) q[4];
rz(1.4289158815826504) q[4];
ry(-1.6055388912491184) q[5];
rz(-2.6272448493214817) q[5];
ry(1.9892145006285489) q[6];
rz(1.4559691174624294) q[6];
ry(0.7885051409412303) q[7];
rz(2.14628851554701) q[7];
ry(-0.30108567237971773) q[8];
rz(1.8066341447320742) q[8];
ry(-2.9922878373273227) q[9];
rz(3.0242919917601805) q[9];
ry(-0.2956293895722226) q[10];
rz(-0.38486280496132347) q[10];
ry(-2.172798739403304) q[11];
rz(-1.3322861761994131) q[11];
ry(2.9899864954752644) q[12];
rz(1.513736096340891) q[12];
ry(-0.2642979129460622) q[13];
rz(-0.09038621655793477) q[13];
ry(-0.3947321683287955) q[14];
rz(-3.055747873084633) q[14];
ry(-1.8369909732093517) q[15];
rz(2.8190769529162143) q[15];
ry(-0.04037987312283888) q[16];
rz(-0.6743642078055997) q[16];
ry(-3.099075352466417) q[17];
rz(-0.4109423863784954) q[17];
ry(-2.445005264046654) q[18];
rz(1.69644659314541) q[18];
ry(0.7218199220983648) q[19];
rz(-0.806322191959981) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.957738781664845) q[0];
rz(-1.146391571178361) q[0];
ry(1.7634431971501339) q[1];
rz(-2.8075837270592316) q[1];
ry(-0.012787392995557683) q[2];
rz(-0.41289311408054363) q[2];
ry(-1.5717947786694362) q[3];
rz(1.1375211420345195) q[3];
ry(-3.066979031548135) q[4];
rz(-2.6382199818353005) q[4];
ry(2.7160475929183843) q[5];
rz(0.7631148968910312) q[5];
ry(2.705481294719991) q[6];
rz(-1.4099940585599606) q[6];
ry(-0.10057820332880141) q[7];
rz(2.5371072468683074) q[7];
ry(-2.0659208989485065) q[8];
rz(1.6960391411305595) q[8];
ry(-0.21709048761477767) q[9];
rz(3.0501667065140445) q[9];
ry(-0.5722518949930039) q[10];
rz(2.9707553028786) q[10];
ry(-3.1248210432641446) q[11];
rz(-1.0790249493692983) q[11];
ry(-2.3606992435673604) q[12];
rz(-1.1285478320934974) q[12];
ry(-2.4680558130831733) q[13];
rz(2.443930558329733) q[13];
ry(0.8885212394297435) q[14];
rz(0.43720838599621364) q[14];
ry(-1.8284422117150798) q[15];
rz(0.7566433863565216) q[15];
ry(-0.3226265222457487) q[16];
rz(0.32523645428964887) q[16];
ry(-0.7005954386731812) q[17];
rz(1.2451294162480613) q[17];
ry(2.361547033583053) q[18];
rz(-0.48268264688007445) q[18];
ry(-2.5476603103791056) q[19];
rz(2.4089600161920863) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.6852932010343036) q[0];
rz(-1.7135838484942083) q[0];
ry(0.16791790805825) q[1];
rz(-0.8165339335155676) q[1];
ry(-1.5705387244699685) q[2];
rz(1.488368932804652) q[2];
ry(-1.539926081686069) q[3];
rz(-1.4251488109119395) q[3];
ry(-3.0245694420441347) q[4];
rz(2.938070478954306) q[4];
ry(-1.3288689323318632) q[5];
rz(-0.8757948651064414) q[5];
ry(-2.063784973008101) q[6];
rz(2.262461623858943) q[6];
ry(1.009963355234168) q[7];
rz(-1.0201140786742249) q[7];
ry(0.1762771097507202) q[8];
rz(-0.646276070140693) q[8];
ry(3.0643249005969144) q[9];
rz(-0.4701387543563245) q[9];
ry(2.116755655133879) q[10];
rz(-0.0043449772311522855) q[10];
ry(-1.649624720414466) q[11];
rz(3.045280202231499) q[11];
ry(0.39483854861475415) q[12];
rz(-1.987854483093668) q[12];
ry(-0.04771243642342249) q[13];
rz(-2.443603151131219) q[13];
ry(1.6465312607715457) q[14];
rz(1.513459498350108) q[14];
ry(-0.001521030211382168) q[15];
rz(-1.459709318544955) q[15];
ry(1.5598980185883196) q[16];
rz(-2.6142864579867826) q[16];
ry(0.02804559794310621) q[17];
rz(-2.784948287317669) q[17];
ry(-0.9237746591859582) q[18];
rz(-0.03697911223275854) q[18];
ry(-0.31585849320145876) q[19];
rz(-2.080996960548539) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.3225145664022473) q[0];
rz(-1.8202556056992865) q[0];
ry(-1.5705259707152128) q[1];
rz(2.4100526167392395) q[1];
ry(-2.5563809647711095) q[2];
rz(-0.12300159615684028) q[2];
ry(1.7764965338628995) q[3];
rz(1.2900325858471904) q[3];
ry(1.7609016374969269) q[4];
rz(0.19543859079925952) q[4];
ry(1.5990130462584684) q[5];
rz(2.8551162117615214) q[5];
ry(0.6693758440270683) q[6];
rz(-2.5756439665713144) q[6];
ry(-0.689099053251577) q[7];
rz(0.7895687624533219) q[7];
ry(-2.521294240663482) q[8];
rz(-0.6157634777892821) q[8];
ry(1.0941025147177053) q[9];
rz(-0.9920246049107994) q[9];
ry(0.24867043961306323) q[10];
rz(2.72500592502806) q[10];
ry(-1.5668402173441978) q[11];
rz(0.24826066531399513) q[11];
ry(-2.1896596372155965) q[12];
rz(-2.5441227665930954) q[12];
ry(-0.1258512394065736) q[13];
rz(-0.12145129744738446) q[13];
ry(-1.7701744072671655) q[14];
rz(3.092705701037489) q[14];
ry(0.0018749418465376527) q[15];
rz(-1.572975117594168) q[15];
ry(-2.944704914136383) q[16];
rz(2.404143371471628) q[16];
ry(-1.6116354890739233) q[17];
rz(3.104809917606407) q[17];
ry(1.957422271709384) q[18];
rz(2.1476370784908196) q[18];
ry(0.6459953363507906) q[19];
rz(0.21923044456509458) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5695571037575917) q[0];
rz(-0.3490985381382945) q[0];
ry(3.0896325264165267) q[1];
rz(0.27277141896119955) q[1];
ry(-1.577717024447419) q[2];
rz(1.6749516325058147) q[2];
ry(2.5243962149022505) q[3];
rz(1.4106341975444139) q[3];
ry(3.0170533416600147) q[4];
rz(-1.1929671616123512) q[4];
ry(1.6814474059308333) q[5];
rz(-0.3467582603859611) q[5];
ry(0.2746554398938157) q[6];
rz(2.8494393729944023) q[6];
ry(-2.681211629092711) q[7];
rz(-1.7975762803102746) q[7];
ry(-0.021712859260541312) q[8];
rz(1.664317850270209) q[8];
ry(6.445508486502326e-05) q[9];
rz(2.477528023800143) q[9];
ry(3.137744084595914) q[10];
rz(-3.052725485694777) q[10];
ry(2.5792724118268398) q[11];
rz(2.978384583832123) q[11];
ry(-1.5583472453943183) q[12];
rz(-3.1261349509359277) q[12];
ry(-0.18655062165554548) q[13];
rz(1.5577125330349306) q[13];
ry(0.43801263340397) q[14];
rz(2.470324437357378) q[14];
ry(-0.001275545564945081) q[15];
rz(1.6410147271787032) q[15];
ry(-0.5653091602629654) q[16];
rz(-0.36055244500640526) q[16];
ry(-2.873693339889233) q[17];
rz(-2.6336196376195247) q[17];
ry(-1.577377347087384) q[18];
rz(2.835251863380982) q[18];
ry(-0.8998782695914919) q[19];
rz(0.8953768253512703) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.780075856486735) q[0];
rz(1.729404452220173) q[0];
ry(-0.10893699063609909) q[1];
rz(1.9005535971505152) q[1];
ry(0.5266650936314292) q[2];
rz(-0.13588736901901513) q[2];
ry(-1.331518185020591) q[3];
rz(0.8390876776069166) q[3];
ry(-0.03716043689798365) q[4];
rz(-1.8315280830868446) q[4];
ry(-1.7224250216449437) q[5];
rz(0.5328565662050408) q[5];
ry(-1.7421560737902737) q[6];
rz(0.0031435930862260264) q[6];
ry(-0.8745808694159516) q[7];
rz(-1.813367056768756) q[7];
ry(-1.3914970891706373) q[8];
rz(-2.796770941723054) q[8];
ry(1.8525754430156236) q[9];
rz(-0.33912536112443314) q[9];
ry(-3.0690626926167934) q[10];
rz(-2.0050636704693714) q[10];
ry(-1.2725475096230277) q[11];
rz(-1.4320170943810355) q[11];
ry(-0.3513835929127346) q[12];
rz(-0.8065004478985467) q[12];
ry(1.566078767028959) q[13];
rz(1.4291676483789173) q[13];
ry(-2.8137134079093022) q[14];
rz(-2.9346936303309987) q[14];
ry(3.1408431662904346) q[15];
rz(2.6267431901921783) q[15];
ry(1.4740172966042895) q[16];
rz(-0.27258043522509734) q[16];
ry(3.1392262721993083) q[17];
rz(-1.0742621182114058) q[17];
ry(-1.608915553738182) q[18];
rz(-1.5827819915843186) q[18];
ry(-1.2686617680564716) q[19];
rz(-0.8084531670349593) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.001258948066862331) q[0];
rz(-1.831807471632252) q[0];
ry(-3.079760530300967) q[1];
rz(1.3290018470595695) q[1];
ry(-0.6567268000008818) q[2];
rz(2.325852637830611) q[2];
ry(0.003404644126950802) q[3];
rz(2.348005794690314) q[3];
ry(-0.002820165130787977) q[4];
rz(-1.2491481091056222) q[4];
ry(1.1342348264750943) q[5];
rz(1.875204883834649) q[5];
ry(-1.8451595749503424) q[6];
rz(1.5437736233855337) q[6];
ry(1.515639934200263) q[7];
rz(0.5540571035883476) q[7];
ry(2.8032278063356655) q[8];
rz(-2.2009461133488317) q[8];
ry(3.0503694392584944) q[9];
rz(2.1045065123866635) q[9];
ry(-3.1206867386464636) q[10];
rz(-2.9928450984701875) q[10];
ry(-1.53437367606945) q[11];
rz(1.4372342307835053) q[11];
ry(-3.116671611358155) q[12];
rz(-2.398578959543862) q[12];
ry(-3.136524130225203) q[13];
rz(1.2818223992080056) q[13];
ry(1.5653569477988223) q[14];
rz(1.5898762322503153) q[14];
ry(3.139913666565178) q[15];
rz(1.0572657400878613) q[15];
ry(-2.9134687256995613) q[16];
rz(1.0376130469594287) q[16];
ry(1.5654998106344484) q[17];
rz(1.3448435570945452) q[17];
ry(1.6493858544406668) q[18];
rz(-3.119184722809488) q[18];
ry(-3.1114447391567737) q[19];
rz(0.5044282965352606) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.1354060996939355) q[0];
rz(-1.7960240286707136) q[0];
ry(1.543827945097619) q[1];
rz(-1.7107534905562245) q[1];
ry(-0.7149263434767317) q[2];
rz(0.3721412718689513) q[2];
ry(2.0038886072813185) q[3];
rz(-2.54432595037501) q[3];
ry(-3.1321027255629437) q[4];
rz(1.774454996715967) q[4];
ry(-2.9312315814986865) q[5];
rz(-2.639026535517) q[5];
ry(-2.799770449817001) q[6];
rz(-1.5418762692270993) q[6];
ry(-3.121220995200322) q[7];
rz(-1.7521359271704897) q[7];
ry(0.20313705299607765) q[8];
rz(2.2013433255267607) q[8];
ry(0.29128060443131915) q[9];
rz(0.36581967210976973) q[9];
ry(-3.1284464282514413) q[10];
rz(1.932003373355965) q[10];
ry(-1.399549333378567) q[11];
rz(-1.9200656830220044) q[11];
ry(1.5406605368629815) q[12];
rz(3.06458107663343) q[12];
ry(-1.592466074549642) q[13];
rz(-1.5578146967787176) q[13];
ry(-1.5309493466056177) q[14];
rz(1.4701119808982934) q[14];
ry(0.2815013669690971) q[15];
rz(1.7369225569379068) q[15];
ry(-1.217831502509399) q[16];
rz(0.01870715443470683) q[16];
ry(1.5301603474177472) q[17];
rz(3.111009976417015) q[17];
ry(-1.5704834388092639) q[18];
rz(-1.6084118103140714) q[18];
ry(-2.6537082892179624) q[19];
rz(3.0064483782703886) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.1353412830928145) q[0];
rz(2.6734377093008623) q[0];
ry(1.5956898251758025) q[1];
rz(-0.13635234929912263) q[1];
ry(1.0132671369451636) q[2];
rz(0.44246753269299255) q[2];
ry(-0.6187272102182862) q[3];
rz(-0.6750647263419838) q[3];
ry(-3.106730202233793) q[4];
rz(1.5057487728319598) q[4];
ry(2.4270072146593438) q[5];
rz(-2.3987743623497115) q[5];
ry(-2.053410280153857) q[6];
rz(-2.545158358931487) q[6];
ry(-2.1894792761557618) q[7];
rz(-2.2564284262017726) q[7];
ry(-0.0645987308217304) q[8];
rz(-3.099835805337569) q[8];
ry(-0.9626923470483011) q[9];
rz(2.815531454152005) q[9];
ry(1.7260765033690488) q[10];
rz(-0.9846423674453346) q[10];
ry(1.6139257787472507) q[11];
rz(-2.018801731761365) q[11];
ry(-0.043623103280710396) q[12];
rz(-1.4717448188589632) q[12];
ry(3.072738944392664) q[13];
rz(-0.2657945377261187) q[13];
ry(-1.5024675420921076) q[14];
rz(0.6518586238515258) q[14];
ry(0.24451867827972212) q[15];
rz(-1.6621930606541644) q[15];
ry(-1.4565821170845228) q[16];
rz(-1.5075484245895758) q[16];
ry(0.0994521793604139) q[17];
rz(1.5178503335739864) q[17];
ry(0.05499477858665043) q[18];
rz(1.9019111636898371) q[18];
ry(1.4271127394612027) q[19];
rz(-0.7355381723229173) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.5659826889916109) q[0];
rz(-2.155547491252574) q[0];
ry(-3.0456359353618465) q[1];
rz(-1.7174029630650551) q[1];
ry(0.106910073440897) q[2];
rz(-3.068127446646937) q[2];
ry(-0.23980687517377178) q[3];
rz(3.0220246774945414) q[3];
ry(-1.582448736972899) q[4];
rz(-2.4069015892767998) q[4];
ry(-0.9909216661430404) q[5];
rz(1.4323216601029811) q[5];
ry(3.1001199779869393) q[6];
rz(0.1408424442657859) q[6];
ry(0.2814414771507156) q[7];
rz(-2.9898396051642977) q[7];
ry(3.114650652794402) q[8];
rz(-0.07076605930430101) q[8];
ry(-3.1040956351519022) q[9];
rz(2.6893469305786875) q[9];
ry(-3.0856732924300005) q[10];
rz(-0.9905761578300102) q[10];
ry(0.039176298674412025) q[11];
rz(-2.6764228479462195) q[11];
ry(1.521507979381397) q[12];
rz(2.395538870374714) q[12];
ry(-1.715950137956626) q[13];
rz(-2.5988125255732824) q[13];
ry(0.01068800209610457) q[14];
rz(2.339296281740484) q[14];
ry(0.0008498922550552733) q[15];
rz(0.3717284523824221) q[15];
ry(-1.289635445824515) q[16];
rz(1.3084709677270743) q[16];
ry(1.6330246718946795) q[17];
rz(-0.5964503314401839) q[17];
ry(-0.026796366225370075) q[18];
rz(1.2056661871354795) q[18];
ry(-1.2401570831836661) q[19];
rz(-2.4053998341519653) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.132111110738178) q[0];
rz(1.4203747012439445) q[0];
ry(-1.5564586517680787) q[1];
rz(1.3878135062223365) q[1];
ry(-2.586123122998298) q[2];
rz(-2.6269562109026077) q[2];
ry(1.5451112191408676) q[3];
rz(0.4044247682459812) q[3];
ry(-3.138023767768526) q[4];
rz(1.8021640385836792) q[4];
ry(3.0937189591434713) q[5];
rz(0.8288109444985511) q[5];
ry(-0.312233995315502) q[6];
rz(1.6461436146984074) q[6];
ry(-1.9488417738816146) q[7];
rz(3.048833511138495) q[7];
ry(2.8082193561289204) q[8];
rz(-1.0088633809544696) q[8];
ry(1.0487852508250735) q[9];
rz(-1.4779625922072757) q[9];
ry(-1.4008935092433576) q[10];
rz(-2.6667797311491674) q[10];
ry(1.203137174052152) q[11];
rz(-1.7702915825372791) q[11];
ry(0.023131129535604564) q[12];
rz(-0.47492697467162076) q[12];
ry(3.132159884721504) q[13];
rz(-1.4895217372153562) q[13];
ry(-0.39981667861213044) q[14];
rz(0.3767381594818561) q[14];
ry(2.9018804546168484) q[15];
rz(1.9571701095640126) q[15];
ry(1.6421841731560365) q[16];
rz(-1.5524319739573533) q[16];
ry(-0.055406623534447696) q[17];
rz(-2.5411167794287848) q[17];
ry(-1.574042730372811) q[18];
rz(-0.40891300745641884) q[18];
ry(2.786972849493324) q[19];
rz(-0.7730982881582683) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.1142840804302674) q[0];
rz(0.4380079732114562) q[0];
ry(1.6439900518151382) q[1];
rz(1.5378004606413342) q[1];
ry(1.5701448542509708) q[2];
rz(-2.783959998263853) q[2];
ry(-2.319997143743366) q[3];
rz(0.7256282702766939) q[3];
ry(-2.687873709765403) q[4];
rz(-2.021241838731154) q[4];
ry(-2.47364403825559) q[5];
rz(0.10596630895931991) q[5];
ry(-1.2190826757166118) q[6];
rz(1.381512955933383) q[6];
ry(-3.0266483062155647) q[7];
rz(2.643791427476993) q[7];
ry(-0.6573593148800495) q[8];
rz(1.1200893470533242) q[8];
ry(-1.4725725793077156) q[9];
rz(0.7752143068012671) q[9];
ry(-2.138813544039329) q[10];
rz(-1.4689426578695377) q[10];
ry(1.7111809988303497) q[11];
rz(0.34577758091233773) q[11];
ry(-2.927097017095611) q[12];
rz(1.0885588183878048) q[12];
ry(0.33240799080081107) q[13];
rz(-0.31251649190783937) q[13];
ry(-1.568735579071309) q[14];
rz(1.5701205438981651) q[14];
ry(-0.00034034103632851753) q[15];
rz(-1.2141214064068993) q[15];
ry(1.7956076077667413) q[16];
rz(-1.5774493190465995) q[16];
ry(-1.5711306785126968) q[17];
rz(-0.0927580381826636) q[17];
ry(-0.02695166670328586) q[18];
rz(3.1047367167715176) q[18];
ry(3.133716293141965) q[19];
rz(0.3272443841877122) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.00894585731249836) q[0];
rz(-0.23585938229128267) q[0];
ry(1.568663731608261) q[1];
rz(-0.7785312559070001) q[1];
ry(0.007166342697166427) q[2];
rz(-0.8068165028251755) q[2];
ry(-0.0031530420118665668) q[3];
rz(-3.0412247245148967) q[3];
ry(-0.03003817895946259) q[4];
rz(2.880326475762496) q[4];
ry(-0.03668281846041868) q[5];
rz(-0.2640222353424216) q[5];
ry(3.002329980378123) q[6];
rz(2.5470128027227172) q[6];
ry(-0.18670909758295196) q[7];
rz(1.2633621063138722) q[7];
ry(-3.060836753787757) q[8];
rz(-0.19540984445812074) q[8];
ry(0.009153636155866557) q[9];
rz(-1.1290603954027212) q[9];
ry(-3.06494965423677) q[10];
rz(-3.0294327808462937) q[10];
ry(0.04543063087662798) q[11];
rz(-2.150843489114295) q[11];
ry(-0.005568109892564467) q[12];
rz(3.0233594521894664) q[12];
ry(3.107795401574807) q[13];
rz(2.5497934845375143) q[13];
ry(1.567648595355696) q[14];
rz(1.1629696489844994) q[14];
ry(3.1395936267360063) q[15];
rz(2.051682962018951) q[15];
ry(-1.581569730665807) q[16];
rz(1.6065273553929378) q[16];
ry(-3.083699446605395) q[17];
rz(-2.8836265023393164) q[17];
ry(-0.010586930782875089) q[18];
rz(-3.0946308961976583) q[18];
ry(1.4064620985308658) q[19];
rz(-1.798237164331855) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.1363102892539738) q[0];
rz(0.8790969518396112) q[0];
ry(0.6859418061168271) q[1];
rz(-0.03647084540033373) q[1];
ry(-1.8499337635972057) q[2];
rz(2.305749289951968) q[2];
ry(-1.3590909223123615) q[3];
rz(-1.1321720734447511) q[3];
ry(0.37125038477841626) q[4];
rz(-0.7120984749555602) q[4];
ry(-1.4380756106282213) q[5];
rz(-1.0656944787049953) q[5];
ry(1.2982675291925085) q[6];
rz(-2.863707765442734) q[6];
ry(-2.9998516413821354) q[7];
rz(2.714306589499461) q[7];
ry(-2.683534966451502) q[8];
rz(0.8667450528738136) q[8];
ry(1.3019934232132901) q[9];
rz(1.2481299222870357) q[9];
ry(1.871619797220128) q[10];
rz(2.2222528370502737) q[10];
ry(-0.13534596643748742) q[11];
rz(0.11096664972340343) q[11];
ry(-1.2971253479666283) q[12];
rz(3.1039042516758832) q[12];
ry(1.2207276041420023) q[13];
rz(-0.14906135453070699) q[13];
ry(-2.7870420047784497) q[14];
rz(2.2275338888675824) q[14];
ry(1.2145728832198603) q[15];
rz(-0.10313803613930882) q[15];
ry(2.71382191892354) q[16];
rz(-0.2097977791526375) q[16];
ry(1.611964100299078) q[17];
rz(-1.6623129605697757) q[17];
ry(1.3676984689702663) q[18];
rz(1.3978451329717512) q[18];
ry(1.8204254555660717) q[19];
rz(1.3533877418687514) q[19];