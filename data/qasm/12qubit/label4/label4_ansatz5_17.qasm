OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.2082624982062313) q[0];
ry(-1.9505610922669014) q[1];
cx q[0],q[1];
ry(-0.8055508087248464) q[0];
ry(-0.5100868580703406) q[1];
cx q[0],q[1];
ry(-2.4030138953757687) q[2];
ry(1.2733318383227923) q[3];
cx q[2],q[3];
ry(2.3622819764887084) q[2];
ry(-0.15448580916934798) q[3];
cx q[2],q[3];
ry(-2.440238515727423) q[4];
ry(1.182939114585844) q[5];
cx q[4],q[5];
ry(-0.6600788710715263) q[4];
ry(1.250440807063027) q[5];
cx q[4],q[5];
ry(-2.4777119600296533) q[6];
ry(-2.5129822660060452) q[7];
cx q[6],q[7];
ry(2.4184315703111756) q[6];
ry(-1.9363205172682312) q[7];
cx q[6],q[7];
ry(1.3838233337542203) q[8];
ry(1.4639630573811466) q[9];
cx q[8],q[9];
ry(-2.638505038494036) q[8];
ry(0.03416394224488173) q[9];
cx q[8],q[9];
ry(-2.7939627463074093) q[10];
ry(0.9959742138147636) q[11];
cx q[10],q[11];
ry(0.28591143781350364) q[10];
ry(2.9827974830042727) q[11];
cx q[10],q[11];
ry(-3.001554978177118) q[1];
ry(-1.9634779683422598) q[2];
cx q[1],q[2];
ry(-0.3258257294290061) q[1];
ry(-1.8281072959702296) q[2];
cx q[1],q[2];
ry(-1.3829320975073307) q[3];
ry(-2.6114857621416157) q[4];
cx q[3],q[4];
ry(-0.5563355305627615) q[3];
ry(-1.9630276603458237) q[4];
cx q[3],q[4];
ry(2.604076309321817) q[5];
ry(2.4738892768618994) q[6];
cx q[5],q[6];
ry(-1.550795228305903) q[5];
ry(0.06441168010343289) q[6];
cx q[5],q[6];
ry(1.1310100808772772) q[7];
ry(2.041270336026123) q[8];
cx q[7],q[8];
ry(0.4213155074493017) q[7];
ry(-3.0717400287430947) q[8];
cx q[7],q[8];
ry(3.1178473889024336) q[9];
ry(-1.4508053886879084) q[10];
cx q[9],q[10];
ry(2.6859094418088114) q[9];
ry(-0.9609101492917601) q[10];
cx q[9],q[10];
ry(1.8529974750018694) q[0];
ry(2.15393674327633) q[1];
cx q[0],q[1];
ry(-0.5915629509712765) q[0];
ry(1.2332301745996106) q[1];
cx q[0],q[1];
ry(2.0079572511365544) q[2];
ry(0.30041833275830854) q[3];
cx q[2],q[3];
ry(1.8526560798686578) q[2];
ry(-2.886959572744581) q[3];
cx q[2],q[3];
ry(0.17545878260143483) q[4];
ry(0.04645941245192642) q[5];
cx q[4],q[5];
ry(-0.012157030925014746) q[4];
ry(-1.704384237560621) q[5];
cx q[4],q[5];
ry(1.737277999577758) q[6];
ry(-2.8499842878175032) q[7];
cx q[6],q[7];
ry(2.212186948465436) q[6];
ry(0.9789803323051993) q[7];
cx q[6],q[7];
ry(-2.5436352556860964) q[8];
ry(1.2217500352461395) q[9];
cx q[8],q[9];
ry(-2.3562285091972193) q[8];
ry(-1.871539291110759) q[9];
cx q[8],q[9];
ry(-2.8877586317106028) q[10];
ry(2.1871305460581505) q[11];
cx q[10],q[11];
ry(-1.911933460063592) q[10];
ry(-1.6720696152583292) q[11];
cx q[10],q[11];
ry(0.8261470010342151) q[1];
ry(-0.3112613581244684) q[2];
cx q[1],q[2];
ry(3.0604929268046996) q[1];
ry(1.7781625043358593) q[2];
cx q[1],q[2];
ry(-3.106512212361858) q[3];
ry(1.8789657100765815) q[4];
cx q[3],q[4];
ry(0.874554065695209) q[3];
ry(0.6010401599396803) q[4];
cx q[3],q[4];
ry(-2.011238651664846) q[5];
ry(1.9155686548626791) q[6];
cx q[5],q[6];
ry(3.1008398255253407) q[5];
ry(0.021614839583787848) q[6];
cx q[5],q[6];
ry(2.9394400570467782) q[7];
ry(-1.176836823108676) q[8];
cx q[7],q[8];
ry(0.14876709857257084) q[7];
ry(-1.4998008826635634) q[8];
cx q[7],q[8];
ry(-2.275327406359465) q[9];
ry(3.1300341149167155) q[10];
cx q[9],q[10];
ry(-1.8989176806771635) q[9];
ry(-1.8027357250132447) q[10];
cx q[9],q[10];
ry(1.812418891914451) q[0];
ry(0.9354354026366911) q[1];
cx q[0],q[1];
ry(-3.028665759761019) q[0];
ry(1.8064341472301413) q[1];
cx q[0],q[1];
ry(-1.0851542540365622) q[2];
ry(1.394210305966527) q[3];
cx q[2],q[3];
ry(1.061944476631087) q[2];
ry(2.1371324666089864) q[3];
cx q[2],q[3];
ry(2.376383434873106) q[4];
ry(2.075375636293109) q[5];
cx q[4],q[5];
ry(-0.01165197770762827) q[4];
ry(0.29410783389360606) q[5];
cx q[4],q[5];
ry(-0.2931797732681214) q[6];
ry(1.8514928958462455) q[7];
cx q[6],q[7];
ry(3.00691361551261) q[6];
ry(-3.1364759655641454) q[7];
cx q[6],q[7];
ry(-2.8803536892680452) q[8];
ry(-0.6844013554086303) q[9];
cx q[8],q[9];
ry(-1.4815012662406402) q[8];
ry(-1.7884178050651744) q[9];
cx q[8],q[9];
ry(3.0256813748963927) q[10];
ry(-2.2368916963073864) q[11];
cx q[10],q[11];
ry(2.2141505051049846) q[10];
ry(-2.300392205936912) q[11];
cx q[10],q[11];
ry(0.29147766019082244) q[1];
ry(1.6710124701836697) q[2];
cx q[1],q[2];
ry(0.42679418477636233) q[1];
ry(0.24276625152026288) q[2];
cx q[1],q[2];
ry(2.6102307134950786) q[3];
ry(1.325076860826083) q[4];
cx q[3],q[4];
ry(2.873352848001144) q[3];
ry(2.4472845483239096) q[4];
cx q[3],q[4];
ry(-2.0160485811736963) q[5];
ry(-2.3701906670326607) q[6];
cx q[5],q[6];
ry(-3.0146878298654873) q[5];
ry(0.03215171685503205) q[6];
cx q[5],q[6];
ry(-3.0093702226066075) q[7];
ry(2.53259058911525) q[8];
cx q[7],q[8];
ry(3.07742480345173) q[7];
ry(0.2549019752810481) q[8];
cx q[7],q[8];
ry(-0.9989948959456303) q[9];
ry(2.9482656482605676) q[10];
cx q[9],q[10];
ry(-0.5601187932477263) q[9];
ry(-1.6253495065687031) q[10];
cx q[9],q[10];
ry(2.7820230333654585) q[0];
ry(-1.941670474608328) q[1];
cx q[0],q[1];
ry(-1.714455263486269) q[0];
ry(2.210130307087607) q[1];
cx q[0],q[1];
ry(0.5031554740288617) q[2];
ry(1.1392321489668011) q[3];
cx q[2],q[3];
ry(0.4841050317776005) q[2];
ry(0.0024256067821955413) q[3];
cx q[2],q[3];
ry(-1.1092687375728603) q[4];
ry(-3.1357842246823804) q[5];
cx q[4],q[5];
ry(3.1394196388614932) q[4];
ry(-0.4126266690082563) q[5];
cx q[4],q[5];
ry(0.3040857094705983) q[6];
ry(-2.767092618414572) q[7];
cx q[6],q[7];
ry(0.935270697499206) q[6];
ry(2.775215546992357) q[7];
cx q[6],q[7];
ry(-1.966672662042943) q[8];
ry(2.0803437214324427) q[9];
cx q[8],q[9];
ry(-0.45707343066545825) q[8];
ry(1.497603476909571) q[9];
cx q[8],q[9];
ry(-1.2658953605982521) q[10];
ry(-2.0514090051322786) q[11];
cx q[10],q[11];
ry(1.088354116023674) q[10];
ry(0.7842667633143714) q[11];
cx q[10],q[11];
ry(-1.1726344185158704) q[1];
ry(2.6493173015951217) q[2];
cx q[1],q[2];
ry(-3.0477369541101114) q[1];
ry(0.030145733406808633) q[2];
cx q[1],q[2];
ry(-0.024736081170600777) q[3];
ry(-0.5759756600066375) q[4];
cx q[3],q[4];
ry(3.0719239153359514) q[3];
ry(-2.6426242714279793) q[4];
cx q[3],q[4];
ry(-2.0064854301380493) q[5];
ry(-0.27058741290370847) q[6];
cx q[5],q[6];
ry(0.700548397301926) q[5];
ry(-3.1224043281844898) q[6];
cx q[5],q[6];
ry(-0.42697197988430013) q[7];
ry(0.07788653896104769) q[8];
cx q[7],q[8];
ry(1.5021088383188277) q[7];
ry(2.5250757161383692) q[8];
cx q[7],q[8];
ry(-0.10181232460698307) q[9];
ry(-2.579961895884696) q[10];
cx q[9],q[10];
ry(1.9651560621624975) q[9];
ry(0.7304259667823167) q[10];
cx q[9],q[10];
ry(1.7171111668457972) q[0];
ry(-0.8141791248788188) q[1];
cx q[0],q[1];
ry(-2.5122703939338997) q[0];
ry(0.8264484697923393) q[1];
cx q[0],q[1];
ry(0.28814193357784235) q[2];
ry(1.252209678462023) q[3];
cx q[2],q[3];
ry(-0.6919102428925101) q[2];
ry(-2.389838968665737) q[3];
cx q[2],q[3];
ry(-0.6341791594861378) q[4];
ry(0.7647605696744556) q[5];
cx q[4],q[5];
ry(3.14048131141867) q[4];
ry(2.9245519546602554) q[5];
cx q[4],q[5];
ry(-1.570794264810942) q[6];
ry(-1.8026135494849647) q[7];
cx q[6],q[7];
ry(2.988268181423448) q[6];
ry(0.1537828839226818) q[7];
cx q[6],q[7];
ry(-0.6112256604775395) q[8];
ry(1.1204585158005005) q[9];
cx q[8],q[9];
ry(-0.44628399276782366) q[8];
ry(-2.4301893669229044) q[9];
cx q[8],q[9];
ry(-0.05601972542623112) q[10];
ry(-0.2014936310910418) q[11];
cx q[10],q[11];
ry(1.7005685599774556) q[10];
ry(-2.6071368999766924) q[11];
cx q[10],q[11];
ry(-2.657527734120026) q[1];
ry(-0.9854545072438246) q[2];
cx q[1],q[2];
ry(-2.58385038568939) q[1];
ry(1.5154956507523794) q[2];
cx q[1],q[2];
ry(1.6368356118440153) q[3];
ry(0.20325043196040157) q[4];
cx q[3],q[4];
ry(-0.927578133922144) q[3];
ry(-1.1258599377526668) q[4];
cx q[3],q[4];
ry(2.8254179809215194) q[5];
ry(-2.4104905695598813) q[6];
cx q[5],q[6];
ry(1.0767979916550245) q[5];
ry(-3.119883100301968) q[6];
cx q[5],q[6];
ry(-2.1438110962372328) q[7];
ry(0.8183966100670883) q[8];
cx q[7],q[8];
ry(-3.135148935515421) q[7];
ry(1.69498247934954) q[8];
cx q[7],q[8];
ry(1.355450438119354) q[9];
ry(1.1672508877700238) q[10];
cx q[9],q[10];
ry(0.9339648109521715) q[9];
ry(1.9685822225640774) q[10];
cx q[9],q[10];
ry(-2.883927124779847) q[0];
ry(-2.8300495750006687) q[1];
cx q[0],q[1];
ry(2.532262728121896) q[0];
ry(-0.051870789060073266) q[1];
cx q[0],q[1];
ry(0.6777838410174852) q[2];
ry(2.756163592767061) q[3];
cx q[2],q[3];
ry(2.82831016153665) q[2];
ry(1.825189221769737) q[3];
cx q[2],q[3];
ry(0.19668419764497624) q[4];
ry(-2.935346617043521) q[5];
cx q[4],q[5];
ry(-3.1304740878380066) q[4];
ry(-1.7880187000334713) q[5];
cx q[4],q[5];
ry(1.5117551257274517) q[6];
ry(-2.465794859550472) q[7];
cx q[6],q[7];
ry(3.086799925548643) q[6];
ry(3.0398387234764797) q[7];
cx q[6],q[7];
ry(2.4440802226020706) q[8];
ry(2.7279196930699134) q[9];
cx q[8],q[9];
ry(1.972149216021521) q[8];
ry(2.4342906181553436) q[9];
cx q[8],q[9];
ry(-2.0181111471016226) q[10];
ry(3.0287338152204666) q[11];
cx q[10],q[11];
ry(0.5353897223297197) q[10];
ry(2.586192045248984) q[11];
cx q[10],q[11];
ry(0.7777082454314144) q[1];
ry(-2.388028279823562) q[2];
cx q[1],q[2];
ry(2.1752903720101706) q[1];
ry(3.118240187462895) q[2];
cx q[1],q[2];
ry(-1.148641054703143) q[3];
ry(2.952695058254981) q[4];
cx q[3],q[4];
ry(-0.9295558839972818) q[3];
ry(-2.1051718815162492) q[4];
cx q[3],q[4];
ry(3.087764122162297) q[5];
ry(-1.508645692234765) q[6];
cx q[5],q[6];
ry(-1.4844684981481346) q[5];
ry(0.0073206460794335015) q[6];
cx q[5],q[6];
ry(-1.059784205728331) q[7];
ry(-1.55966221697122) q[8];
cx q[7],q[8];
ry(-1.7420212756152798) q[7];
ry(-1.3860548996793773) q[8];
cx q[7],q[8];
ry(-2.5645589524645778) q[9];
ry(0.2362040586998262) q[10];
cx q[9],q[10];
ry(-0.2072811343315366) q[9];
ry(-0.8773302823425801) q[10];
cx q[9],q[10];
ry(1.9205292326498242) q[0];
ry(-0.17108855103933981) q[1];
cx q[0],q[1];
ry(0.07055268846540987) q[0];
ry(2.010168416594876) q[1];
cx q[0],q[1];
ry(-2.542154143408997) q[2];
ry(1.4043709737141814) q[3];
cx q[2],q[3];
ry(-2.80628909706909) q[2];
ry(-3.084697963373187) q[3];
cx q[2],q[3];
ry(1.0379658452169032) q[4];
ry(-2.97974798337136) q[5];
cx q[4],q[5];
ry(-0.9285081785566154) q[4];
ry(2.409596484534051) q[5];
cx q[4],q[5];
ry(1.8163660091369787) q[6];
ry(-0.7520964439275579) q[7];
cx q[6],q[7];
ry(0.09662702685958813) q[6];
ry(-0.04555852169215822) q[7];
cx q[6],q[7];
ry(2.9533663722709687) q[8];
ry(-2.8932706188244084) q[9];
cx q[8],q[9];
ry(-0.8197379884851923) q[8];
ry(-1.3752352948784685) q[9];
cx q[8],q[9];
ry(0.6865588296788454) q[10];
ry(-0.6720831163585097) q[11];
cx q[10],q[11];
ry(3.0431037113448545) q[10];
ry(0.7002740052091784) q[11];
cx q[10],q[11];
ry(-3.11446746178806) q[1];
ry(1.1766015288061666) q[2];
cx q[1],q[2];
ry(1.0886995531529449) q[1];
ry(2.8314070413650625) q[2];
cx q[1],q[2];
ry(0.18500400062341357) q[3];
ry(-2.8046667222720183) q[4];
cx q[3],q[4];
ry(-0.001903602836100582) q[3];
ry(2.1025084660586444) q[4];
cx q[3],q[4];
ry(1.2444871609733772) q[5];
ry(-2.4759714621580238) q[6];
cx q[5],q[6];
ry(1.1963542732971204) q[5];
ry(3.1377767666855014) q[6];
cx q[5],q[6];
ry(0.43436252166198747) q[7];
ry(1.0116140204348196) q[8];
cx q[7],q[8];
ry(-2.8380652747402193) q[7];
ry(-1.52514895013518) q[8];
cx q[7],q[8];
ry(0.7998379864891764) q[9];
ry(-1.855922625231973) q[10];
cx q[9],q[10];
ry(-1.1421422998203428) q[9];
ry(2.201093726021227) q[10];
cx q[9],q[10];
ry(-3.006407247485695) q[0];
ry(2.7007552509959507) q[1];
cx q[0],q[1];
ry(2.923492057093403) q[0];
ry(-1.8839268420058017) q[1];
cx q[0],q[1];
ry(1.31413916215736) q[2];
ry(2.6539981321006554) q[3];
cx q[2],q[3];
ry(-2.0865471224469196) q[2];
ry(-1.284342865339556) q[3];
cx q[2],q[3];
ry(1.8466974648954804) q[4];
ry(-1.9157290298346596) q[5];
cx q[4],q[5];
ry(-1.108797716272961) q[4];
ry(1.3508467267763296) q[5];
cx q[4],q[5];
ry(3.1248800746335146) q[6];
ry(-1.5363037716169101) q[7];
cx q[6],q[7];
ry(1.8738606925536496) q[6];
ry(2.848715357267518) q[7];
cx q[6],q[7];
ry(0.6765979827551347) q[8];
ry(-2.113618293519213) q[9];
cx q[8],q[9];
ry(-0.0779538011316685) q[8];
ry(1.9329587646225397) q[9];
cx q[8],q[9];
ry(-0.40165981850859556) q[10];
ry(1.162625780637967) q[11];
cx q[10],q[11];
ry(2.7383803235618447) q[10];
ry(-0.9687183532511492) q[11];
cx q[10],q[11];
ry(-2.769269117806134) q[1];
ry(1.4392750471456734) q[2];
cx q[1],q[2];
ry(0.2066597132883456) q[1];
ry(1.810409992543831) q[2];
cx q[1],q[2];
ry(0.18917555530817068) q[3];
ry(3.06914523095535) q[4];
cx q[3],q[4];
ry(0.007530535638354152) q[3];
ry(2.741371062643006) q[4];
cx q[3],q[4];
ry(1.6196108816089225) q[5];
ry(-1.893432110821328) q[6];
cx q[5],q[6];
ry(0.15453866040446407) q[5];
ry(-3.1412041522645997) q[6];
cx q[5],q[6];
ry(3.0824122574595347) q[7];
ry(-0.9259232628640666) q[8];
cx q[7],q[8];
ry(0.19645568396641977) q[7];
ry(0.07800797254179587) q[8];
cx q[7],q[8];
ry(0.4413035614909715) q[9];
ry(2.23192525598074) q[10];
cx q[9],q[10];
ry(-1.7288204850488906) q[9];
ry(1.6123549196588796) q[10];
cx q[9],q[10];
ry(-1.263756637825518) q[0];
ry(0.7584132301587792) q[1];
cx q[0],q[1];
ry(-1.5455521400776044) q[0];
ry(2.6592459285714876) q[1];
cx q[0],q[1];
ry(2.991901637691415) q[2];
ry(-2.715165425597888) q[3];
cx q[2],q[3];
ry(-1.1415751129034124) q[2];
ry(-2.737846787725144) q[3];
cx q[2],q[3];
ry(2.554933586720613) q[4];
ry(-2.6159684402225523) q[5];
cx q[4],q[5];
ry(-0.03212050944684642) q[4];
ry(0.33595103669488147) q[5];
cx q[4],q[5];
ry(-1.2860754214944399) q[6];
ry(3.0623404439105784) q[7];
cx q[6],q[7];
ry(-2.0659520225768633) q[6];
ry(-2.715377825015312) q[7];
cx q[6],q[7];
ry(-1.576209012455573) q[8];
ry(-1.2510752309045117) q[9];
cx q[8],q[9];
ry(-1.7104884862023335) q[8];
ry(-2.815233710147874) q[9];
cx q[8],q[9];
ry(-2.5862447204773678) q[10];
ry(0.32623879705520764) q[11];
cx q[10],q[11];
ry(1.8892374182794647) q[10];
ry(1.3254360482027545) q[11];
cx q[10],q[11];
ry(2.1881130659432806) q[1];
ry(0.8057579241215125) q[2];
cx q[1],q[2];
ry(2.7334981634877726) q[1];
ry(0.8743869555053854) q[2];
cx q[1],q[2];
ry(0.947600542498586) q[3];
ry(-2.5936345319753253) q[4];
cx q[3],q[4];
ry(-2.570353969971279) q[3];
ry(2.000991975432817) q[4];
cx q[3],q[4];
ry(0.15210654910848811) q[5];
ry(-2.1171371748258627) q[6];
cx q[5],q[6];
ry(1.8677133654507285) q[5];
ry(-3.130508402336459) q[6];
cx q[5],q[6];
ry(-1.2788826259031514) q[7];
ry(2.3590497210724624) q[8];
cx q[7],q[8];
ry(1.211036773761687) q[7];
ry(2.1777881658335563) q[8];
cx q[7],q[8];
ry(-2.3657878921174293) q[9];
ry(2.866708752975928) q[10];
cx q[9],q[10];
ry(-0.3060742533006779) q[9];
ry(-1.6040423826927876) q[10];
cx q[9],q[10];
ry(-2.7549737425652605) q[0];
ry(0.10821577994920163) q[1];
cx q[0],q[1];
ry(-1.9642233847704986) q[0];
ry(1.7581598686344684) q[1];
cx q[0],q[1];
ry(-2.813610736558608) q[2];
ry(2.6940913177924304) q[3];
cx q[2],q[3];
ry(-2.853655775809542) q[2];
ry(-0.6839372912336872) q[3];
cx q[2],q[3];
ry(-1.7615625785883342) q[4];
ry(-0.9576704315212234) q[5];
cx q[4],q[5];
ry(-2.2800515177031135) q[4];
ry(-2.8856082634627827) q[5];
cx q[4],q[5];
ry(0.7233606915216351) q[6];
ry(-1.4673460277306942) q[7];
cx q[6],q[7];
ry(-0.2869969097646801) q[6];
ry(0.7675010477342682) q[7];
cx q[6],q[7];
ry(-2.5893767163813446) q[8];
ry(-1.3888053416817348) q[9];
cx q[8],q[9];
ry(2.1046624433820886) q[8];
ry(-0.73395782174026) q[9];
cx q[8],q[9];
ry(-0.09482234957305799) q[10];
ry(-0.7757909457072375) q[11];
cx q[10],q[11];
ry(-0.22627035456794875) q[10];
ry(-2.4750597375063554) q[11];
cx q[10],q[11];
ry(-2.0877935773556775) q[1];
ry(1.2648305335553935) q[2];
cx q[1],q[2];
ry(1.6818546382910207) q[1];
ry(-2.6411465052927405) q[2];
cx q[1],q[2];
ry(-1.1757220715639072) q[3];
ry(2.115938528935485) q[4];
cx q[3],q[4];
ry(-0.005681444413076986) q[3];
ry(2.0309139133002905) q[4];
cx q[3],q[4];
ry(-2.463818054083787) q[5];
ry(0.8837853059249654) q[6];
cx q[5],q[6];
ry(3.133179444241939) q[5];
ry(-3.1382950685718236) q[6];
cx q[5],q[6];
ry(0.27052240646172354) q[7];
ry(-1.9227457672286363) q[8];
cx q[7],q[8];
ry(-0.9796205736919684) q[7];
ry(3.029584382132761) q[8];
cx q[7],q[8];
ry(1.6721100812270253) q[9];
ry(-1.411485315910448) q[10];
cx q[9],q[10];
ry(2.0397300686882094) q[9];
ry(1.8953524010744864) q[10];
cx q[9],q[10];
ry(1.124722412960311) q[0];
ry(0.4668019964993845) q[1];
cx q[0],q[1];
ry(2.500149232685155) q[0];
ry(2.240347504119737) q[1];
cx q[0],q[1];
ry(-0.41934564226861903) q[2];
ry(-0.7509784845645696) q[3];
cx q[2],q[3];
ry(-2.548265555740439) q[2];
ry(2.5048860250258986) q[3];
cx q[2],q[3];
ry(-2.31340865418862) q[4];
ry(-2.169281815253953) q[5];
cx q[4],q[5];
ry(-2.234562841737566) q[4];
ry(-2.269765949210768) q[5];
cx q[4],q[5];
ry(-1.7129566955807707) q[6];
ry(1.6441178196918589) q[7];
cx q[6],q[7];
ry(0.8286015463223324) q[6];
ry(-0.542973820579928) q[7];
cx q[6],q[7];
ry(-2.0760470955451105) q[8];
ry(1.3425439899305829) q[9];
cx q[8],q[9];
ry(-1.2464770337091444) q[8];
ry(-2.007675369793626) q[9];
cx q[8],q[9];
ry(-2.298599753291829) q[10];
ry(-2.243503726897575) q[11];
cx q[10],q[11];
ry(2.519377267974431) q[10];
ry(0.5515577084191632) q[11];
cx q[10],q[11];
ry(1.2145282287825636) q[1];
ry(-3.0128513774846257) q[2];
cx q[1],q[2];
ry(1.7912174103191574) q[1];
ry(-1.294418334226446) q[2];
cx q[1],q[2];
ry(-1.520700645012338) q[3];
ry(1.807385636536429) q[4];
cx q[3],q[4];
ry(3.1165093229254577) q[3];
ry(-0.15813803665956644) q[4];
cx q[3],q[4];
ry(-1.6002181282998553) q[5];
ry(1.7107442507107145) q[6];
cx q[5],q[6];
ry(-3.1412712944602967) q[5];
ry(-0.00739648758149997) q[6];
cx q[5],q[6];
ry(0.7394429425802228) q[7];
ry(0.6710913909989421) q[8];
cx q[7],q[8];
ry(0.06060716658136058) q[7];
ry(-3.087423917120107) q[8];
cx q[7],q[8];
ry(2.979935415527997) q[9];
ry(1.1748496689093726) q[10];
cx q[9],q[10];
ry(1.2196285660199697) q[9];
ry(0.8792476843927767) q[10];
cx q[9],q[10];
ry(0.3568985897365154) q[0];
ry(0.7021244812085908) q[1];
cx q[0],q[1];
ry(-2.646332964355723) q[0];
ry(1.273893106232103) q[1];
cx q[0],q[1];
ry(2.1132785256992275) q[2];
ry(0.7502081543139112) q[3];
cx q[2],q[3];
ry(0.6766379844614756) q[2];
ry(-1.889556971479303) q[3];
cx q[2],q[3];
ry(1.4916886562081375) q[4];
ry(1.987775989671463) q[5];
cx q[4],q[5];
ry(2.8039291094600967) q[4];
ry(2.9931133817310323) q[5];
cx q[4],q[5];
ry(2.270078097836798) q[6];
ry(-1.6080826354629787) q[7];
cx q[6],q[7];
ry(2.4715594898250184) q[6];
ry(-0.7832548685840363) q[7];
cx q[6],q[7];
ry(-3.129900959340239) q[8];
ry(1.7470033971253374) q[9];
cx q[8],q[9];
ry(-0.2832246136558805) q[8];
ry(-2.2341380581531816) q[9];
cx q[8],q[9];
ry(-0.53219787871069) q[10];
ry(-0.48198232386276896) q[11];
cx q[10],q[11];
ry(0.9261877493436383) q[10];
ry(-2.7313431809079645) q[11];
cx q[10],q[11];
ry(-0.7805466531200206) q[1];
ry(-2.757077126988204) q[2];
cx q[1],q[2];
ry(-2.394190005240247) q[1];
ry(1.7620306543493833) q[2];
cx q[1],q[2];
ry(-2.990938127999368) q[3];
ry(1.5851780454566171) q[4];
cx q[3],q[4];
ry(-3.129918712866534) q[3];
ry(-2.900385512680807) q[4];
cx q[3],q[4];
ry(-1.3444247119922879) q[5];
ry(-0.8867032420399266) q[6];
cx q[5],q[6];
ry(0.005612828556357163) q[5];
ry(-0.04181809378784074) q[6];
cx q[5],q[6];
ry(1.9556056311894334) q[7];
ry(-0.33229158982362406) q[8];
cx q[7],q[8];
ry(-2.1780444458296033) q[7];
ry(-0.5203086032334667) q[8];
cx q[7],q[8];
ry(2.317717764932046) q[9];
ry(-1.1037052052261793) q[10];
cx q[9],q[10];
ry(2.47096761056658) q[9];
ry(1.84102559808949) q[10];
cx q[9],q[10];
ry(0.26397460029677067) q[0];
ry(2.1780722924619624) q[1];
cx q[0],q[1];
ry(-0.22591869320458594) q[0];
ry(-0.7730341266645163) q[1];
cx q[0],q[1];
ry(-1.0367669800052721) q[2];
ry(1.415367616155832) q[3];
cx q[2],q[3];
ry(-2.1605835428011373) q[2];
ry(0.352340441068721) q[3];
cx q[2],q[3];
ry(-0.8702524504294553) q[4];
ry(1.499313511245483) q[5];
cx q[4],q[5];
ry(1.4464048916085388) q[4];
ry(-2.216716960693076) q[5];
cx q[4],q[5];
ry(-2.7791511322685714) q[6];
ry(-1.257332964119862) q[7];
cx q[6],q[7];
ry(-2.561707299588833) q[6];
ry(-0.145433612744565) q[7];
cx q[6],q[7];
ry(-1.3093008058236109) q[8];
ry(-0.034438326676894775) q[9];
cx q[8],q[9];
ry(-1.6278232251979423) q[8];
ry(0.16905004122287678) q[9];
cx q[8],q[9];
ry(3.0033131895807856) q[10];
ry(2.3102880796111296) q[11];
cx q[10],q[11];
ry(2.131769647266168) q[10];
ry(-1.2995341873556752) q[11];
cx q[10],q[11];
ry(-1.163109352881432) q[1];
ry(0.7639128018815532) q[2];
cx q[1],q[2];
ry(-1.0042958497897967) q[1];
ry(2.6544886549110696) q[2];
cx q[1],q[2];
ry(-0.6448381672016695) q[3];
ry(-2.947471190361706) q[4];
cx q[3],q[4];
ry(0.018341023225844744) q[3];
ry(-0.0807866490536579) q[4];
cx q[3],q[4];
ry(2.5045349661677325) q[5];
ry(-1.3019244036004016) q[6];
cx q[5],q[6];
ry(1.6096574912072845) q[5];
ry(-3.1332567885755522) q[6];
cx q[5],q[6];
ry(-2.787791575957445) q[7];
ry(2.4536803508474416) q[8];
cx q[7],q[8];
ry(-0.8416374699574414) q[7];
ry(-2.87431104004042) q[8];
cx q[7],q[8];
ry(0.5777281251966131) q[9];
ry(-0.7493319501931213) q[10];
cx q[9],q[10];
ry(1.862228707901413) q[9];
ry(-2.263913364742785) q[10];
cx q[9],q[10];
ry(2.3775618052128507) q[0];
ry(-0.9157893237045185) q[1];
cx q[0],q[1];
ry(-0.3808906803583527) q[0];
ry(2.4312502688124025) q[1];
cx q[0],q[1];
ry(-0.10438282373390462) q[2];
ry(-0.8157681636391324) q[3];
cx q[2],q[3];
ry(0.6092725737076423) q[2];
ry(-0.9069530493483855) q[3];
cx q[2],q[3];
ry(2.0190968857628007) q[4];
ry(-0.43805036880670767) q[5];
cx q[4],q[5];
ry(-1.5945330127159558) q[4];
ry(0.061348269011470606) q[5];
cx q[4],q[5];
ry(-1.4286319351026386) q[6];
ry(-2.5309617704456095) q[7];
cx q[6],q[7];
ry(2.9320922740123345) q[6];
ry(0.6434509762892597) q[7];
cx q[6],q[7];
ry(-0.006248132356182785) q[8];
ry(0.7470578722798926) q[9];
cx q[8],q[9];
ry(-0.927913000161542) q[8];
ry(-2.312737326434413) q[9];
cx q[8],q[9];
ry(-0.5323943393911241) q[10];
ry(-0.4791051867876061) q[11];
cx q[10],q[11];
ry(2.5472699946014234) q[10];
ry(-0.24084765092360402) q[11];
cx q[10],q[11];
ry(-0.1482124278585419) q[1];
ry(1.7684194629751113) q[2];
cx q[1],q[2];
ry(1.2644654176735628) q[1];
ry(0.7080729130557141) q[2];
cx q[1],q[2];
ry(-3.0429897519421836) q[3];
ry(2.8860507776201842) q[4];
cx q[3],q[4];
ry(-1.349053233209724) q[3];
ry(1.7271253955275767) q[4];
cx q[3],q[4];
ry(1.1851100660539904) q[5];
ry(1.6639636135012603) q[6];
cx q[5],q[6];
ry(0.08334902404299042) q[5];
ry(0.04924835874078035) q[6];
cx q[5],q[6];
ry(-2.636092513912808) q[7];
ry(0.5286047091431606) q[8];
cx q[7],q[8];
ry(2.040985826720668) q[7];
ry(2.6880134073866673) q[8];
cx q[7],q[8];
ry(-2.741113711838616) q[9];
ry(2.204153486308508) q[10];
cx q[9],q[10];
ry(-2.55684997339779) q[9];
ry(-3.0423910655501087) q[10];
cx q[9],q[10];
ry(-0.2617603337011607) q[0];
ry(1.2120286454197968) q[1];
cx q[0],q[1];
ry(-2.296161352611693) q[0];
ry(0.8272339659988095) q[1];
cx q[0],q[1];
ry(2.9628170653507175) q[2];
ry(-0.8802228313191698) q[3];
cx q[2],q[3];
ry(0.099303429449213) q[2];
ry(2.939140473680678) q[3];
cx q[2],q[3];
ry(0.11122313337709322) q[4];
ry(-1.2808984258064542) q[5];
cx q[4],q[5];
ry(-0.2031378125631731) q[4];
ry(3.1283868367849044) q[5];
cx q[4],q[5];
ry(3.110287330289336) q[6];
ry(-0.9342284788969959) q[7];
cx q[6],q[7];
ry(-0.06320740132969149) q[6];
ry(0.013953183974870775) q[7];
cx q[6],q[7];
ry(0.037633814650126496) q[8];
ry(2.9911324035586997) q[9];
cx q[8],q[9];
ry(-1.2233499796796208) q[8];
ry(-0.03560038662989084) q[9];
cx q[8],q[9];
ry(0.4805014044480924) q[10];
ry(0.6097168361825847) q[11];
cx q[10],q[11];
ry(2.223916101561123) q[10];
ry(2.472230360236211) q[11];
cx q[10],q[11];
ry(1.7725637963074181) q[1];
ry(-2.7761768035859578) q[2];
cx q[1],q[2];
ry(-2.3014864823271592) q[1];
ry(-0.19661010236411605) q[2];
cx q[1],q[2];
ry(2.5273599893335104) q[3];
ry(-1.1924399889387745) q[4];
cx q[3],q[4];
ry(-1.1545968709618286) q[3];
ry(1.8694185027133523) q[4];
cx q[3],q[4];
ry(0.18163493242250417) q[5];
ry(1.402491747014197) q[6];
cx q[5],q[6];
ry(-0.01745474140740713) q[5];
ry(-0.009108213893317311) q[6];
cx q[5],q[6];
ry(1.2500985903879678) q[7];
ry(2.5132036130517452) q[8];
cx q[7],q[8];
ry(0.7963866008401159) q[7];
ry(-2.945544680596863) q[8];
cx q[7],q[8];
ry(-3.037527460001187) q[9];
ry(1.205068229725084) q[10];
cx q[9],q[10];
ry(1.7166120782274357) q[9];
ry(-1.2758502215295646) q[10];
cx q[9],q[10];
ry(-3.0930718888959468) q[0];
ry(-1.2132495670871606) q[1];
cx q[0],q[1];
ry(-0.19663431912623433) q[0];
ry(-1.8415137115630067) q[1];
cx q[0],q[1];
ry(0.4216081036008354) q[2];
ry(1.1025834466876994) q[3];
cx q[2],q[3];
ry(-0.12795321165313675) q[2];
ry(3.0982909654062825) q[3];
cx q[2],q[3];
ry(-0.7527845579314114) q[4];
ry(1.6020851769137263) q[5];
cx q[4],q[5];
ry(-0.04039106824544463) q[4];
ry(0.018104536262611236) q[5];
cx q[4],q[5];
ry(2.564749142507383) q[6];
ry(2.2957673891314005) q[7];
cx q[6],q[7];
ry(-0.08046178627558387) q[6];
ry(3.0752132346281065) q[7];
cx q[6],q[7];
ry(-2.806620684617932) q[8];
ry(1.9355175369646425) q[9];
cx q[8],q[9];
ry(-1.0429369487861528) q[8];
ry(-0.44616364919950185) q[9];
cx q[8],q[9];
ry(-1.1976465466084347) q[10];
ry(1.7097296425588586) q[11];
cx q[10],q[11];
ry(-2.6899114364899575) q[10];
ry(-0.9368576309028096) q[11];
cx q[10],q[11];
ry(-2.886879007543937) q[1];
ry(-2.8080373337543305) q[2];
cx q[1],q[2];
ry(2.375764781705217) q[1];
ry(2.4080487894928573) q[2];
cx q[1],q[2];
ry(0.4212948832497066) q[3];
ry(0.5015491074660078) q[4];
cx q[3],q[4];
ry(-2.3174366205773733) q[3];
ry(-1.455118395457312) q[4];
cx q[3],q[4];
ry(3.000898961469286) q[5];
ry(0.8693079374888879) q[6];
cx q[5],q[6];
ry(-2.7755201546981403) q[5];
ry(0.98781783344534) q[6];
cx q[5],q[6];
ry(-0.6851214536905238) q[7];
ry(1.2240992309464742) q[8];
cx q[7],q[8];
ry(-2.0362079725755198) q[7];
ry(-2.876710162592405) q[8];
cx q[7],q[8];
ry(0.18781099158649717) q[9];
ry(-2.0781566735773125) q[10];
cx q[9],q[10];
ry(-2.1551546499837864) q[9];
ry(1.1819981980166965) q[10];
cx q[9],q[10];
ry(0.37404267768445415) q[0];
ry(1.0051399997164892) q[1];
cx q[0],q[1];
ry(1.267356037607169) q[0];
ry(0.05953985363817348) q[1];
cx q[0],q[1];
ry(-3.123206259107483) q[2];
ry(-0.6033199697711278) q[3];
cx q[2],q[3];
ry(0.13791784691336062) q[2];
ry(1.220853032432923) q[3];
cx q[2],q[3];
ry(-0.33727417336213805) q[4];
ry(0.504240699545599) q[5];
cx q[4],q[5];
ry(0.01988117353771113) q[4];
ry(3.1269284504276915) q[5];
cx q[4],q[5];
ry(-1.6424636618641482) q[6];
ry(1.65947097275627) q[7];
cx q[6],q[7];
ry(3.133740765645919) q[6];
ry(1.3487498439613412) q[7];
cx q[6],q[7];
ry(-1.306564605517651) q[8];
ry(-0.7276967577767373) q[9];
cx q[8],q[9];
ry(1.9689995688591369) q[8];
ry(-2.910134581942195) q[9];
cx q[8],q[9];
ry(-0.3558118655711544) q[10];
ry(0.7651969072173269) q[11];
cx q[10],q[11];
ry(1.962530608138679) q[10];
ry(-1.7805688456360587) q[11];
cx q[10],q[11];
ry(-2.415397479613763) q[1];
ry(1.450830330939393) q[2];
cx q[1],q[2];
ry(-1.6678719415299508) q[1];
ry(0.1110104213619989) q[2];
cx q[1],q[2];
ry(-1.5240808259797078) q[3];
ry(2.641237601835215) q[4];
cx q[3],q[4];
ry(-2.8951899138003023) q[3];
ry(0.9188218388507394) q[4];
cx q[3],q[4];
ry(-2.353356822472566) q[5];
ry(-2.7223163090263576) q[6];
cx q[5],q[6];
ry(0.010525333845897045) q[5];
ry(-1.0387552926448693) q[6];
cx q[5],q[6];
ry(-1.7652757377383415) q[7];
ry(-2.2442876223052863) q[8];
cx q[7],q[8];
ry(-1.4319080495391798) q[7];
ry(0.00016767357850084536) q[8];
cx q[7],q[8];
ry(-1.2463798067897915) q[9];
ry(-2.1734756971209253) q[10];
cx q[9],q[10];
ry(1.8053449198625924) q[9];
ry(0.8776107714411524) q[10];
cx q[9],q[10];
ry(1.8837010662492335) q[0];
ry(0.11354469894926224) q[1];
cx q[0],q[1];
ry(-1.9606891517872778) q[0];
ry(-2.0199595953827756) q[1];
cx q[0],q[1];
ry(0.5286030006583244) q[2];
ry(1.65722587074203) q[3];
cx q[2],q[3];
ry(-1.5714504955902697) q[2];
ry(3.081710250957151) q[3];
cx q[2],q[3];
ry(1.2168983750970055) q[4];
ry(0.3061262007732637) q[5];
cx q[4],q[5];
ry(-3.1398697095550574) q[4];
ry(-0.000598203388856966) q[5];
cx q[4],q[5];
ry(2.495449924867632) q[6];
ry(2.997289569974584) q[7];
cx q[6],q[7];
ry(3.1395655338358215) q[6];
ry(0.008880445626872067) q[7];
cx q[6],q[7];
ry(2.556710139835514) q[8];
ry(-1.1077785935350244) q[9];
cx q[8],q[9];
ry(2.157138650495613) q[8];
ry(-0.6451797376960169) q[9];
cx q[8],q[9];
ry(1.9051160245090364) q[10];
ry(-1.6263246131612301) q[11];
cx q[10],q[11];
ry(0.9918538425143306) q[10];
ry(-0.8426762044734268) q[11];
cx q[10],q[11];
ry(-3.051442222007045) q[1];
ry(-2.0132331851269694) q[2];
cx q[1],q[2];
ry(2.6501241114865772) q[1];
ry(-0.9490511297571357) q[2];
cx q[1],q[2];
ry(-1.5928457523510673) q[3];
ry(-1.922189124331117) q[4];
cx q[3],q[4];
ry(3.1167102783628526) q[3];
ry(2.2980457175459477) q[4];
cx q[3],q[4];
ry(0.2660124071672076) q[5];
ry(-1.3292789505020886) q[6];
cx q[5],q[6];
ry(0.45787960199164957) q[5];
ry(2.1124761005725654) q[6];
cx q[5],q[6];
ry(-3.1073357695310864) q[7];
ry(-1.1643109191364944) q[8];
cx q[7],q[8];
ry(-1.2991300705650608) q[7];
ry(-0.20371810214433927) q[8];
cx q[7],q[8];
ry(0.2052860125827922) q[9];
ry(-2.5730727954594146) q[10];
cx q[9],q[10];
ry(0.795381811363323) q[9];
ry(-1.1463281301081603) q[10];
cx q[9],q[10];
ry(-2.989279492981218) q[0];
ry(-2.7199341448506194) q[1];
cx q[0],q[1];
ry(-0.19828262605502722) q[0];
ry(-0.8356448299756493) q[1];
cx q[0],q[1];
ry(2.0109185622784134) q[2];
ry(-1.5661987193333375) q[3];
cx q[2],q[3];
ry(0.9961758594354793) q[2];
ry(-0.1804869298476769) q[3];
cx q[2],q[3];
ry(-0.7188644624205632) q[4];
ry(0.3441809637975002) q[5];
cx q[4],q[5];
ry(-2.8961754153073924) q[4];
ry(-0.8130448957869545) q[5];
cx q[4],q[5];
ry(0.03801950653125502) q[6];
ry(1.5931633639749556) q[7];
cx q[6],q[7];
ry(0.11145128994283134) q[6];
ry(-0.004977097130719699) q[7];
cx q[6],q[7];
ry(-0.42936684321955976) q[8];
ry(2.70216104478679) q[9];
cx q[8],q[9];
ry(-2.3955011930780663) q[8];
ry(1.728732384225756) q[9];
cx q[8],q[9];
ry(0.16177060111268743) q[10];
ry(-3.0051292325249723) q[11];
cx q[10],q[11];
ry(0.45018186105377195) q[10];
ry(-0.35848299693526187) q[11];
cx q[10],q[11];
ry(-2.9047490332557824) q[1];
ry(2.1221776607751064) q[2];
cx q[1],q[2];
ry(-2.015791494621918) q[1];
ry(-1.2768768304556952) q[2];
cx q[1],q[2];
ry(1.5024414790300327) q[3];
ry(-1.5725162259938852) q[4];
cx q[3],q[4];
ry(2.0538238451504935) q[3];
ry(0.09654679971078538) q[4];
cx q[3],q[4];
ry(1.6336306059040533) q[5];
ry(-0.05783285096240487) q[6];
cx q[5],q[6];
ry(0.7030295098473982) q[5];
ry(-0.2234178150816665) q[6];
cx q[5],q[6];
ry(2.321822587624445) q[7];
ry(-1.770873189812888) q[8];
cx q[7],q[8];
ry(3.11737140023088) q[7];
ry(-0.005169043968397347) q[8];
cx q[7],q[8];
ry(2.6816810992006137) q[9];
ry(-2.8289437830253235) q[10];
cx q[9],q[10];
ry(1.6914747868695192) q[9];
ry(3.1313478237408674) q[10];
cx q[9],q[10];
ry(2.9835733575226633) q[0];
ry(2.9812164956094676) q[1];
cx q[0],q[1];
ry(-0.06582467077571863) q[0];
ry(1.913887349744411) q[1];
cx q[0],q[1];
ry(-0.9449179435132696) q[2];
ry(1.576677669909386) q[3];
cx q[2],q[3];
ry(1.623916305353296) q[2];
ry(1.2155727058499846) q[3];
cx q[2],q[3];
ry(1.5710746983326187) q[4];
ry(-1.5771298244258065) q[5];
cx q[4],q[5];
ry(-1.6350961803996462) q[4];
ry(-2.346224591986444) q[5];
cx q[4],q[5];
ry(1.581504899108917) q[6];
ry(-2.2790989402551483) q[7];
cx q[6],q[7];
ry(-3.100005504845345) q[6];
ry(1.5088830923652945) q[7];
cx q[6],q[7];
ry(2.541078851254532) q[8];
ry(2.835962387259053) q[9];
cx q[8],q[9];
ry(1.326205445084062) q[8];
ry(0.5549210065624697) q[9];
cx q[8],q[9];
ry(-0.47511873346547423) q[10];
ry(3.137123207613901) q[11];
cx q[10],q[11];
ry(-1.2258623648870512) q[10];
ry(-2.385714099313083) q[11];
cx q[10],q[11];
ry(1.6924418006645485) q[1];
ry(-1.5035316028499974) q[2];
cx q[1],q[2];
ry(-2.887009129645126) q[1];
ry(-0.09859516425268434) q[2];
cx q[1],q[2];
ry(1.5737776571823867) q[3];
ry(-1.6369673876939324) q[4];
cx q[3],q[4];
ry(3.13338566362843) q[3];
ry(1.6067202444475468) q[4];
cx q[3],q[4];
ry(1.567323311855544) q[5];
ry(-1.5866727262502105) q[6];
cx q[5],q[6];
ry(-3.111016857099751) q[5];
ry(-2.2329360488193695) q[6];
cx q[5],q[6];
ry(-1.0474893053596048) q[7];
ry(0.42368283288472486) q[8];
cx q[7],q[8];
ry(-1.5548482472460574) q[7];
ry(-3.1375478233573797) q[8];
cx q[7],q[8];
ry(2.6493739510872456) q[9];
ry(-2.2831144247711896) q[10];
cx q[9],q[10];
ry(1.8839774049701938) q[9];
ry(0.0326949326072) q[10];
cx q[9],q[10];
ry(1.6278287714454205) q[0];
ry(2.96202115161347) q[1];
ry(0.0230409992216644) q[2];
ry(-1.5693930850877902) q[3];
ry(3.0745341704615248) q[4];
ry(1.5673991314222624) q[5];
ry(3.1232266275011162) q[6];
ry(2.0210223320643603) q[7];
ry(0.018112308210259442) q[8];
ry(2.198269403596492) q[9];
ry(-0.49809204751119296) q[10];
ry(3.0666691497567777) q[11];