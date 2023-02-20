OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.1811186519418957) q[0];
rz(-0.15071545054394253) q[0];
ry(2.5518266227614386) q[1];
rz(-2.4635463203127643) q[1];
ry(1.5365412041929871) q[2];
rz(2.105605808736354) q[2];
ry(0.9362541696643669) q[3];
rz(2.7433547936604596) q[3];
ry(-0.1864079334200645) q[4];
rz(-0.29444094540298943) q[4];
ry(-2.9532007324420375) q[5];
rz(1.85032448607765) q[5];
ry(0.0021800059983831943) q[6];
rz(-0.3325916050361073) q[6];
ry(-0.0045259813802278216) q[7];
rz(0.06328603003695614) q[7];
ry(-1.0862548094321711) q[8];
rz(-0.4677813484239345) q[8];
ry(0.37162154442024564) q[9];
rz(1.7357131589626984) q[9];
ry(1.2025829907769838) q[10];
rz(1.3112371852045772) q[10];
ry(2.432719934354233) q[11];
rz(-0.549332507457625) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.8370362413157382) q[0];
rz(-2.8946999183538664) q[0];
ry(1.338316628559219) q[1];
rz(-2.1342909793737714) q[1];
ry(0.5481922505254907) q[2];
rz(-1.0472377332214806) q[2];
ry(-1.8822842682799557) q[3];
rz(-0.4646899588213466) q[3];
ry(2.613207658090298) q[4];
rz(-2.009317410099926) q[4];
ry(-0.5902941220363188) q[5];
rz(2.671488141343706) q[5];
ry(3.1367856430790106) q[6];
rz(-0.33133172239554665) q[6];
ry(-0.00489850110649922) q[7];
rz(2.0491582477899124) q[7];
ry(-1.6792021076921082) q[8];
rz(-1.60151548536132) q[8];
ry(1.491891924420739) q[9];
rz(0.6211466520744847) q[9];
ry(0.15887113828054922) q[10];
rz(-0.9877757015570924) q[10];
ry(-3.0816976266764526) q[11];
rz(-2.429194975270438) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.5892225732578764) q[0];
rz(0.692209558625775) q[0];
ry(-0.4927811422908718) q[1];
rz(-1.8289494638629298) q[1];
ry(2.5892183466437233) q[2];
rz(-0.30626001024355237) q[2];
ry(-1.839801048637162) q[3];
rz(1.3116896194101084) q[3];
ry(-2.7307197648765023) q[4];
rz(-2.244057851753663) q[4];
ry(2.0045527458952566) q[5];
rz(-2.6144957017136434) q[5];
ry(3.1403037910947162) q[6];
rz(-0.6276755219924857) q[6];
ry(1.3120305842997209) q[7];
rz(3.06168024523207) q[7];
ry(-2.548569002439277) q[8];
rz(1.1633120300144775) q[8];
ry(-2.0379312845790483) q[9];
rz(2.7117380007669794) q[9];
ry(-1.402305676271638) q[10];
rz(0.5496146157611754) q[10];
ry(2.177918467361536) q[11];
rz(-2.515075120082892) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.3072072225970883) q[0];
rz(-2.4211544836166747) q[0];
ry(-2.3099410600262) q[1];
rz(2.646878983775068) q[1];
ry(2.2155863956432373) q[2];
rz(0.6466542221079397) q[2];
ry(-0.05501854190925241) q[3];
rz(-1.789289309486509) q[3];
ry(-0.9014001587042139) q[4];
rz(1.6352088349935254) q[4];
ry(-0.002041699300845231) q[5];
rz(1.912916951881262) q[5];
ry(1.8630219052547374) q[6];
rz(1.2441370993289478) q[6];
ry(0.000838656079646371) q[7];
rz(2.570715393954789) q[7];
ry(2.7121524847019622) q[8];
rz(2.9052284038782252) q[8];
ry(0.58891699581722) q[9];
rz(-2.895609066129271) q[9];
ry(-0.293980428838934) q[10];
rz(0.19289899176502434) q[10];
ry(-0.4142501036866175) q[11];
rz(-0.9251749546671362) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.8346233196920664) q[0];
rz(1.2051296921520507) q[0];
ry(0.6176622208621707) q[1];
rz(1.655870779223597) q[1];
ry(-3.107559973281496) q[2];
rz(1.4446510062645619) q[2];
ry(-2.6952133496108637) q[3];
rz(-1.6945406162431518) q[3];
ry(-0.0031497321611642093) q[4];
rz(-1.0884264181936194) q[4];
ry(-0.9419006811847641) q[5];
rz(-1.614130809965829) q[5];
ry(3.133971582299741) q[6];
rz(-0.6750392757084449) q[6];
ry(-0.6659131731597263) q[7];
rz(-2.9789489052236076) q[7];
ry(-2.095427663759839) q[8];
rz(1.4805112989487696) q[8];
ry(1.3661716333117808) q[9];
rz(-2.632769236218662) q[9];
ry(-1.8709243131510753) q[10];
rz(1.3523337149210728) q[10];
ry(-0.4522170302610285) q[11];
rz(-0.5377966699250533) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.9788711088947095) q[0];
rz(1.719161561994448) q[0];
ry(0.23362082963379385) q[1];
rz(0.715765823912399) q[1];
ry(0.001198164163997992) q[2];
rz(0.2077417410178244) q[2];
ry(-1.7818984726237357) q[3];
rz(0.0012259640805627472) q[3];
ry(0.6007556914861744) q[4];
rz(0.9582108803600267) q[4];
ry(-0.0012440381152567254) q[5];
rz(0.6760670537882197) q[5];
ry(1.9306402397640574) q[6];
rz(1.8021503736759765) q[6];
ry(3.140816241266737) q[7];
rz(-1.1972691253180905) q[7];
ry(-2.388881821067349) q[8];
rz(-1.3643940874622267) q[8];
ry(1.3598459461717232) q[9];
rz(0.4345095782371242) q[9];
ry(0.9850910049073731) q[10];
rz(-0.17437687026654886) q[10];
ry(2.097304199197648) q[11];
rz(-0.9434355632942394) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.8030370199718472) q[0];
rz(2.7128655436179185) q[0];
ry(1.4111347386135353) q[1];
rz(-2.1514631595350338) q[1];
ry(-1.955991945651445) q[2];
rz(-2.6270720631410236) q[2];
ry(-2.732561943359924) q[3];
rz(-1.7218808810391695) q[3];
ry(-0.19681571742289877) q[4];
rz(-2.6327684719233084) q[4];
ry(-0.08451157432632511) q[5];
rz(2.2934396644143327) q[5];
ry(-3.139476064082204) q[6];
rz(1.47244821704486) q[6];
ry(3.08499511270207) q[7];
rz(-2.865006088957516) q[7];
ry(-2.9288790278543178) q[8];
rz(2.428707297652317) q[8];
ry(2.3983547594941848) q[9];
rz(1.8147275001156933) q[9];
ry(-2.0656074065689234) q[10];
rz(-2.8361831865846407) q[10];
ry(-2.293681943888399) q[11];
rz(-1.8104178949347967) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.945713361436726) q[0];
rz(-2.083671043118589) q[0];
ry(2.092642018437016) q[1];
rz(-1.1144557169906322) q[1];
ry(1.9345561934281834) q[2];
rz(1.9327562703812484) q[2];
ry(-2.273517917136899) q[3];
rz(1.2983554000646338) q[3];
ry(2.4411338730622143) q[4];
rz(2.428234713117069) q[4];
ry(-3.14121186598397) q[5];
rz(1.7320598581120186) q[5];
ry(3.1409715289618054) q[6];
rz(-1.551144662580323) q[6];
ry(-3.1373664211340238) q[7];
rz(2.778690130329254) q[7];
ry(2.285458525025348) q[8];
rz(0.3005809593306478) q[8];
ry(0.3496656667647836) q[9];
rz(-2.625590485254538) q[9];
ry(-2.1114212225082785) q[10];
rz(2.824394299609712) q[10];
ry(2.5424882380555016) q[11];
rz(-3.085758018981399) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.8401922741395724) q[0];
rz(-2.182730960404044) q[0];
ry(1.6432933916761794) q[1];
rz(-0.2903894818334516) q[1];
ry(-2.7839769664159792) q[2];
rz(3.033126839028194) q[2];
ry(1.2820687919481624) q[3];
rz(2.0421879740677946) q[3];
ry(2.929326405320429) q[4];
rz(0.22315215958367113) q[4];
ry(3.092140154436625) q[5];
rz(-1.701769859325073) q[5];
ry(3.1282389822983236) q[6];
rz(-0.1366638473346896) q[6];
ry(0.3234281753423548) q[7];
rz(-1.8631226009575022) q[7];
ry(0.7301926806771752) q[8];
rz(-0.9940820621231404) q[8];
ry(-2.0652315352623725) q[9];
rz(-0.3278304180454489) q[9];
ry(2.149266681668899) q[10];
rz(2.312502131170741) q[10];
ry(-1.0904670908705578) q[11];
rz(-1.5616710066811992) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.498017355538301) q[0];
rz(1.7465275743251902) q[0];
ry(1.049955220206563) q[1];
rz(-1.9409548349464292) q[1];
ry(2.548106893322351) q[2];
rz(1.6471394939416864) q[2];
ry(-2.214722676775462) q[3];
rz(0.7017379469825067) q[3];
ry(0.27965490018847833) q[4];
rz(1.2032124148669894) q[4];
ry(8.705356702076728e-05) q[5];
rz(1.715380294949754) q[5];
ry(-3.1188264901716725) q[6];
rz(-0.24011810700593195) q[6];
ry(-2.9705069931941477) q[7];
rz(-2.8147105950883664) q[7];
ry(-2.555926564849992) q[8];
rz(-2.6461922938189972) q[8];
ry(1.3198103018360527) q[9];
rz(-0.704814775125528) q[9];
ry(0.6100194253985869) q[10];
rz(1.3864970613359584) q[10];
ry(1.7684527175757907) q[11];
rz(1.2469914264461555) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.656172447058091) q[0];
rz(0.33167853712626005) q[0];
ry(1.8470858039776195) q[1];
rz(1.0411184289705273) q[1];
ry(-0.7840217963368916) q[2];
rz(-0.0016430595391625504) q[2];
ry(-1.7804624920560355) q[3];
rz(-1.3756806169275944) q[3];
ry(-0.3110213359342762) q[4];
rz(1.850448932515209) q[4];
ry(0.012134894853079281) q[5];
rz(0.9398211145130038) q[5];
ry(0.10462897294571062) q[6];
rz(2.7576655517847106) q[6];
ry(-0.18933006924708456) q[7];
rz(0.13598319286954388) q[7];
ry(-1.8451627020626375) q[8];
rz(1.4440653749663168) q[8];
ry(0.3770560299564346) q[9];
rz(-2.8482646259181434) q[9];
ry(3.1008049822938877) q[10];
rz(1.0254062967319053) q[10];
ry(-3.003558310596225) q[11];
rz(2.663143733725491) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.4861456944155784) q[0];
rz(-1.0107063220987247) q[0];
ry(0.11723991113535437) q[1];
rz(1.0776901923135613) q[1];
ry(0.4338766835645444) q[2];
rz(-1.1192329884198662) q[2];
ry(1.2272132756201093) q[3];
rz(1.12390682608105) q[3];
ry(-0.034656003665475055) q[4];
rz(0.9797005144817641) q[4];
ry(-1.3560219738491684e-05) q[5];
rz(1.864478282248986) q[5];
ry(0.00897465422985455) q[6];
rz(0.38028894141026903) q[6];
ry(-3.0424044703701414) q[7];
rz(0.4047581215560951) q[7];
ry(-0.04842851942384015) q[8];
rz(-1.063318731761453) q[8];
ry(0.4616802493163031) q[9];
rz(2.106240733518452) q[9];
ry(2.2122484433145804) q[10];
rz(-1.9916433472858324) q[10];
ry(1.3440536665986444) q[11];
rz(-0.37401517760697467) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.5733198796539476) q[0];
rz(-2.623495937900124) q[0];
ry(-2.3944612643741423) q[1];
rz(-2.4444903362024144) q[1];
ry(0.7086986280401435) q[2];
rz(2.0325405463608046) q[2];
ry(-0.6329495590891313) q[3];
rz(1.5636534597804375) q[3];
ry(2.8597376094687883) q[4];
rz(-2.7102544911915083) q[4];
ry(0.07357195618172785) q[5];
rz(-1.9956151854633468) q[5];
ry(3.0368392722571236) q[6];
rz(-0.7050394921433645) q[6];
ry(0.15724447255616653) q[7];
rz(0.5009484280871526) q[7];
ry(-2.6289271705086965) q[8];
rz(1.701454051797467) q[8];
ry(-1.9729832311369062) q[9];
rz(0.4568943345882674) q[9];
ry(-2.725609921051091) q[10];
rz(-1.5054416710633238) q[10];
ry(0.9962516589745887) q[11];
rz(2.2610756777401693) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.03506431655937418) q[0];
rz(-1.1791342565038225) q[0];
ry(-0.3925136356685939) q[1];
rz(-1.3545273954861485) q[1];
ry(-1.7547688931094862) q[2];
rz(-0.9561099443672351) q[2];
ry(1.215182952584826) q[3];
rz(-0.9195040506114802) q[3];
ry(-2.1087375551430916) q[4];
rz(-0.9228991512820137) q[4];
ry(0.0011009708970476345) q[5];
rz(-0.9878226763463438) q[5];
ry(-0.002759765051455254) q[6];
rz(2.519869745522115) q[6];
ry(0.1276273585368859) q[7];
rz(-1.2698125298391005) q[7];
ry(-0.3660339789970462) q[8];
rz(1.6201994717035888) q[8];
ry(2.564674966626011) q[9];
rz(2.612240434370927) q[9];
ry(-1.5228645398096075) q[10];
rz(1.2960666576681805) q[10];
ry(0.7907085937442925) q[11];
rz(2.155397009919435) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.0477160275410613) q[0];
rz(-0.532341065487708) q[0];
ry(-2.741016677066668) q[1];
rz(1.3655455824611886) q[1];
ry(-3.0229316380100433) q[2];
rz(-1.959852802802735) q[2];
ry(-0.22544862064073) q[3];
rz(1.6852199963973462) q[3];
ry(0.014972427859460069) q[4];
rz(-2.382293859863204) q[4];
ry(0.004656654834273688) q[5];
rz(-3.09461440217205) q[5];
ry(-3.1397746176424466) q[6];
rz(0.913155165833579) q[6];
ry(-2.8097830116936264) q[7];
rz(1.326178339536355) q[7];
ry(1.3751378963091794) q[8];
rz(-0.32980752979962374) q[8];
ry(1.7897157575917664) q[9];
rz(2.5873438257111188) q[9];
ry(-1.619195966324052) q[10];
rz(0.3722237063174738) q[10];
ry(-1.0712800538812013) q[11];
rz(-2.4009830099299725) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.127734649264331) q[0];
rz(0.9485430564945148) q[0];
ry(2.3993841318614173) q[1];
rz(-3.0271013085412575) q[1];
ry(-1.1927709654904308) q[2];
rz(2.769339567989256) q[2];
ry(-2.5919656778649043) q[3];
rz(1.749124598416637) q[3];
ry(-2.2272109786717946) q[4];
rz(-0.3666148779078231) q[4];
ry(-3.1406597960881193) q[5];
rz(2.2620634137445332) q[5];
ry(0.0008675292525987565) q[6];
rz(0.9073282253993469) q[6];
ry(0.030344764878654118) q[7];
rz(2.0117444475248867) q[7];
ry(-0.9425893786218369) q[8];
rz(1.2495905602440693) q[8];
ry(-2.979960883924477) q[9];
rz(0.6132482424578516) q[9];
ry(2.9691102352798633) q[10];
rz(2.874681432032113) q[10];
ry(0.3535727211381934) q[11];
rz(2.6811585084106473) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.6665905771397601) q[0];
rz(-0.3767566717508331) q[0];
ry(-0.8519344324845881) q[1];
rz(2.330263407320392) q[1];
ry(-2.3489470605725216) q[2];
rz(0.0025201074323586425) q[2];
ry(-0.12996612386566575) q[3];
rz(0.30199060914929543) q[3];
ry(2.928505883058649) q[4];
rz(-0.1393172378236401) q[4];
ry(-3.137215336195684) q[5];
rz(0.5852652338228858) q[5];
ry(-3.1251103980718917) q[6];
rz(1.5359537255549522) q[6];
ry(0.41863258415077914) q[7];
rz(1.237353408220883) q[7];
ry(-1.0810793805612668) q[8];
rz(1.011793768787568) q[8];
ry(2.0522508932286794) q[9];
rz(-0.2177525768809545) q[9];
ry(0.16798711544767908) q[10];
rz(0.27115248630397115) q[10];
ry(-1.5516138179165067) q[11];
rz(-0.0433478516742635) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.6807746132863668) q[0];
rz(2.446162015478561) q[0];
ry(-0.9673391179939893) q[1];
rz(0.7695858344144203) q[1];
ry(-2.233480037832602) q[2];
rz(0.4500725535463662) q[2];
ry(-1.1204697803788068) q[3];
rz(-1.1133294426542184) q[3];
ry(2.8267967314266147) q[4];
rz(-2.9652844225519477) q[4];
ry(-1.5660579517435491) q[5];
rz(0.9715152728306778) q[5];
ry(-0.004498492813175048) q[6];
rz(0.2677192703905522) q[6];
ry(1.2326952921772438) q[7];
rz(0.16634439352178013) q[7];
ry(-1.3607623857917384) q[8];
rz(2.1560975745829825) q[8];
ry(-0.574029084945967) q[9];
rz(-1.156873675239839) q[9];
ry(-1.4793527240478135) q[10];
rz(-1.2410896077590676) q[10];
ry(0.1323411379533767) q[11];
rz(-1.0960534101846422) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.9502125472156662) q[0];
rz(1.3977273977680964) q[0];
ry(1.8984151197002772) q[1];
rz(-2.963608703868051) q[1];
ry(-0.4133645414791376) q[2];
rz(-2.8868371329384503) q[2];
ry(-0.0005476288759102163) q[3];
rz(1.8554268106494103) q[3];
ry(1.4329320772629177) q[4];
rz(3.139001815282774) q[4];
ry(3.141355101167025) q[5];
rz(0.9881126648781045) q[5];
ry(-0.0013167498011803502) q[6];
rz(1.520986260211583) q[6];
ry(0.000681484397234945) q[7];
rz(2.9835044968928663) q[7];
ry(1.6616446748415914) q[8];
rz(1.4193619183262878) q[8];
ry(-3.10100598037165) q[9];
rz(1.6741709726065706) q[9];
ry(1.6459222596908407) q[10];
rz(0.344378930815907) q[10];
ry(2.0684137268170604) q[11];
rz(2.7718763617803885) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.318337649805998) q[0];
rz(-1.768977176204758) q[0];
ry(-1.056514287855875) q[1];
rz(2.121692195874955) q[1];
ry(3.1414585722905515) q[2];
rz(2.961055284594187) q[2];
ry(1.7230584758112304) q[3];
rz(-1.5208163692749899) q[3];
ry(1.5424865448848413) q[4];
rz(0.395256811821797) q[4];
ry(3.1026202976274435) q[5];
rz(-0.01649086403233646) q[5];
ry(3.140977817277073) q[6];
rz(-2.4131909081384295) q[6];
ry(1.2359125754529803) q[7];
rz(-0.5192580621848917) q[7];
ry(-2.097738045440881) q[8];
rz(1.594694001364969) q[8];
ry(-1.3245715151776647) q[9];
rz(-0.4493344266658925) q[9];
ry(-0.051329316640046814) q[10];
rz(-0.3182154317945721) q[10];
ry(0.10263173104293788) q[11];
rz(-0.9632609652575654) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.2920903864180964) q[0];
rz(2.088015701529822) q[0];
ry(-1.7964148922046548) q[1];
rz(-2.441827011950771) q[1];
ry(-1.6868164974126114) q[2];
rz(1.4608592121307762) q[2];
ry(-3.138564319006795) q[3];
rz(-2.0448359338262536) q[3];
ry(-2.991703185570651) q[4];
rz(2.6902120536585565) q[4];
ry(0.01335610986939173) q[5];
rz(0.03896556163384002) q[5];
ry(-0.000519453230124256) q[6];
rz(-0.8636730925332216) q[6];
ry(3.1410832234319477) q[7];
rz(-2.567970667562141) q[7];
ry(1.5118908002343723) q[8];
rz(2.5371873998502834) q[8];
ry(-1.647156973226032) q[9];
rz(-0.770489983354155) q[9];
ry(-1.5236925187889598) q[10];
rz(2.5269495109209563) q[10];
ry(1.4565114583062362) q[11];
rz(-0.2595337583819513) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5972810049559234) q[0];
rz(-1.2208206204556777) q[0];
ry(2.530622756334957) q[1];
rz(-0.6721883518269118) q[1];
ry(-0.0003172661350880901) q[2];
rz(0.1127000938892374) q[2];
ry(-3.0989189656031457) q[3];
rz(-2.4426265200897843) q[3];
ry(-3.1411576455054684) q[4];
rz(1.8754536324121434) q[4];
ry(-3.0077494109537093) q[5];
rz(-2.330354095162474) q[5];
ry(-1.5697804226617076) q[6];
rz(0.08034391629458792) q[6];
ry(-1.577838310851278) q[7];
rz(-3.137550483293147) q[7];
ry(0.35581550676519813) q[8];
rz(0.1462099489946447) q[8];
ry(0.45336939215271865) q[9];
rz(-2.2841329293877726) q[9];
ry(0.34220340644589814) q[10];
rz(2.9116117241534143) q[10];
ry(-1.127273116775877) q[11];
rz(0.1322233228943438) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.16950658496457383) q[0];
rz(2.8482834564028496) q[0];
ry(0.18933298981201382) q[1];
rz(2.8315014561441436) q[1];
ry(-1.4900533652769408) q[2];
rz(-1.9613799940756829) q[2];
ry(-1.5735134525818593) q[3];
rz(-3.140472057768549) q[3];
ry(3.140588970373661) q[4];
rz(1.238504545072596) q[4];
ry(-3.140272847756247) q[5];
rz(-2.4581663095515163) q[5];
ry(3.141430625291572) q[6];
rz(-0.5998235838318795) q[6];
ry(3.141384796585403) q[7];
rz(0.09303215530685893) q[7];
ry(1.1354275218810677) q[8];
rz(-2.734451737245504) q[8];
ry(-1.5512457232567334) q[9];
rz(-0.3571331091015496) q[9];
ry(2.097636775527131) q[10];
rz(1.1132309897728547) q[10];
ry(1.9359695424717625) q[11];
rz(-0.0631843661960895) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.526486447387275) q[0];
rz(-0.7695399928260119) q[0];
ry(1.5708068604324368) q[1];
rz(3.140705091236605) q[1];
ry(-1.5649689794055737) q[2];
rz(3.124758525958782) q[2];
ry(-2.6830246712269252) q[3];
rz(1.3521097037441185) q[3];
ry(-3.1359964824338857) q[4];
rz(1.56929249359855) q[4];
ry(0.036628188394350225) q[5];
rz(-0.6595973829520393) q[5];
ry(-3.1387637160472983) q[6];
rz(2.9095991210840055) q[6];
ry(-0.6887571585685004) q[7];
rz(2.6716554700902706) q[7];
ry(3.138684446515338) q[8];
rz(1.1058534125710278) q[8];
ry(-0.0009702083399476535) q[9];
rz(-1.64038198532282) q[9];
ry(-1.0435979290888782) q[10];
rz(2.9543722141579902) q[10];
ry(1.7133074700221043) q[11];
rz(-2.257490895180545) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5715589320575596) q[0];
rz(3.0983458578631415) q[0];
ry(0.7889890728748226) q[1];
rz(3.1260410080278813) q[1];
ry(0.20938121422541275) q[2];
rz(0.01466455949783896) q[2];
ry(3.1408516132037283) q[3];
rz(2.947835581710439) q[3];
ry(3.140913349607982) q[4];
rz(-0.09210214729033238) q[4];
ry(0.0006469409174495416) q[5];
rz(1.3974777775369143) q[5];
ry(3.140782966940946) q[6];
rz(-2.8394321824043836) q[6];
ry(0.00027567661155469716) q[7];
rz(3.0302807911645826) q[7];
ry(-1.2115979723496186) q[8];
rz(0.5697505786135473) q[8];
ry(-3.1382263292034067) q[9];
rz(2.2683068132107054) q[9];
ry(-3.117023159168413) q[10];
rz(1.384241567213932) q[10];
ry(-0.001904587535785415) q[11];
rz(-0.7285372160643223) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.1999461148264431) q[0];
rz(2.193088907166164) q[0];
ry(0.03911728904447731) q[1];
rz(1.7244828066654216) q[1];
ry(1.5702741182943074) q[2];
rz(2.150895870926372) q[2];
ry(3.140927709225196) q[3];
rz(-1.4058754466073224) q[3];
ry(1.4726169726041967) q[4];
rz(2.1458448314117193) q[4];
ry(3.097992441775639) q[5];
rz(0.7523538783760765) q[5];
ry(-3.13712742819638) q[6];
rz(-1.1354581527571102) q[6];
ry(3.016868293891196) q[7];
rz(-1.9141462404546397) q[7];
ry(-0.00019864449064055378) q[8];
rz(1.5022526393034772) q[8];
ry(-3.1399057915022603) q[9];
rz(2.8786348817877156) q[9];
ry(1.5713611801684966) q[10];
rz(0.19379638030063884) q[10];
ry(-2.9641628909316817) q[11];
rz(0.3289300930122812) q[11];