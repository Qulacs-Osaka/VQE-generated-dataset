OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.12172497470508947) q[0];
ry(2.8228972545423545) q[1];
cx q[0],q[1];
ry(2.071984761401673) q[0];
ry(-2.5000215988341314) q[1];
cx q[0],q[1];
ry(2.3008623390046825) q[2];
ry(2.4037901853777233) q[3];
cx q[2],q[3];
ry(-1.4350478185198714) q[2];
ry(2.4935585379868415) q[3];
cx q[2],q[3];
ry(1.1103857798138626) q[4];
ry(0.42596877127017935) q[5];
cx q[4],q[5];
ry(-1.25190169311078) q[4];
ry(-0.9804290699257036) q[5];
cx q[4],q[5];
ry(2.9410933358189966) q[6];
ry(1.6684152672026453) q[7];
cx q[6],q[7];
ry(-1.8027309654375232) q[6];
ry(0.9836586814782153) q[7];
cx q[6],q[7];
ry(-2.623497357176565) q[0];
ry(-2.930717936769932) q[2];
cx q[0],q[2];
ry(1.6091568413648263) q[0];
ry(0.8149267642917852) q[2];
cx q[0],q[2];
ry(0.7605218628475621) q[2];
ry(-1.010300650041816) q[4];
cx q[2],q[4];
ry(-2.8924713392621837) q[2];
ry(2.0847822117070667) q[4];
cx q[2],q[4];
ry(-1.6614778433695458) q[4];
ry(0.7169371454602919) q[6];
cx q[4],q[6];
ry(-1.839005675946268) q[4];
ry(-1.8736761779470392) q[6];
cx q[4],q[6];
ry(-2.654161104386883) q[1];
ry(2.067864743729142) q[3];
cx q[1],q[3];
ry(-0.029517287289293925) q[1];
ry(-1.646447191385463) q[3];
cx q[1],q[3];
ry(-0.6239454653908121) q[3];
ry(2.5096722921473065) q[5];
cx q[3],q[5];
ry(-0.8077675841843099) q[3];
ry(-1.2531335541849196) q[5];
cx q[3],q[5];
ry(-2.8438439157636344) q[5];
ry(2.344081572394022) q[7];
cx q[5],q[7];
ry(0.8304868820959497) q[5];
ry(1.9137418988938137) q[7];
cx q[5],q[7];
ry(-1.1928722140989452) q[0];
ry(2.7845858862263597) q[3];
cx q[0],q[3];
ry(2.0938733421988127) q[0];
ry(-1.0478890900231195) q[3];
cx q[0],q[3];
ry(-0.4732756949099605) q[1];
ry(-1.9431848275718009) q[2];
cx q[1],q[2];
ry(-0.0920902216981192) q[1];
ry(1.9062168297499955) q[2];
cx q[1],q[2];
ry(-2.2955746677933178) q[2];
ry(-2.124201905842102) q[5];
cx q[2],q[5];
ry(1.8267997617615928) q[2];
ry(-1.1967015080812287) q[5];
cx q[2],q[5];
ry(1.100115410727648) q[3];
ry(0.2142071115872186) q[4];
cx q[3],q[4];
ry(1.1907273017748536) q[3];
ry(-1.9079074042986601) q[4];
cx q[3],q[4];
ry(2.086307694209953) q[4];
ry(-0.9777672831112296) q[7];
cx q[4],q[7];
ry(-1.7697598931899152) q[4];
ry(-2.8690050124870994) q[7];
cx q[4],q[7];
ry(1.682007395629971) q[5];
ry(2.885486530346106) q[6];
cx q[5],q[6];
ry(0.14195018269553383) q[5];
ry(0.5842340261040672) q[6];
cx q[5],q[6];
ry(2.634489740160683) q[0];
ry(-3.0924293276538166) q[1];
cx q[0],q[1];
ry(-2.4808945356728014) q[0];
ry(-1.203944243330677) q[1];
cx q[0],q[1];
ry(-2.5053048230472097) q[2];
ry(0.9127272732648435) q[3];
cx q[2],q[3];
ry(-0.13534731494580918) q[2];
ry(1.1347306890967923) q[3];
cx q[2],q[3];
ry(-1.7835303546900854) q[4];
ry(-1.2732787605700273) q[5];
cx q[4],q[5];
ry(-1.9048165064585145) q[4];
ry(1.7095426000222513) q[5];
cx q[4],q[5];
ry(-0.7391275831511868) q[6];
ry(2.922559001251669) q[7];
cx q[6],q[7];
ry(-2.376000711284764) q[6];
ry(-2.6175018841302937) q[7];
cx q[6],q[7];
ry(-1.646958163892413) q[0];
ry(-0.43417201360665114) q[2];
cx q[0],q[2];
ry(2.403733921693071) q[0];
ry(-0.8345488541898902) q[2];
cx q[0],q[2];
ry(0.8330106742424038) q[2];
ry(-2.9085928624543653) q[4];
cx q[2],q[4];
ry(-0.6251423483714743) q[2];
ry(-0.4018584871085915) q[4];
cx q[2],q[4];
ry(1.866905578237053) q[4];
ry(0.34957011725053777) q[6];
cx q[4],q[6];
ry(1.662026720494775) q[4];
ry(0.10325552533126872) q[6];
cx q[4],q[6];
ry(0.4610834028168461) q[1];
ry(-0.8049432853579007) q[3];
cx q[1],q[3];
ry(-1.5870376270048316) q[1];
ry(-2.2802838203862974) q[3];
cx q[1],q[3];
ry(-2.9133616073720106) q[3];
ry(1.2798880013342053) q[5];
cx q[3],q[5];
ry(-0.40900058068129574) q[3];
ry(-3.11259576984401) q[5];
cx q[3],q[5];
ry(2.389778770906821) q[5];
ry(-1.9155463285339165) q[7];
cx q[5],q[7];
ry(1.5352542055622624) q[5];
ry(-0.431775718185901) q[7];
cx q[5],q[7];
ry(-0.22822562277244976) q[0];
ry(2.7948138858579576) q[3];
cx q[0],q[3];
ry(1.4835915614678248) q[0];
ry(0.13077385212505754) q[3];
cx q[0],q[3];
ry(2.4132773853140894) q[1];
ry(-0.17634814734570625) q[2];
cx q[1],q[2];
ry(2.8962912529947804) q[1];
ry(-1.8945108774203339) q[2];
cx q[1],q[2];
ry(1.2053225768181886) q[2];
ry(-0.24032590858728936) q[5];
cx q[2],q[5];
ry(-0.11970868111363737) q[2];
ry(2.0407956731687174) q[5];
cx q[2],q[5];
ry(-0.6508312565117294) q[3];
ry(-0.19941029720494718) q[4];
cx q[3],q[4];
ry(2.0117131027211723) q[3];
ry(2.0769102661893735) q[4];
cx q[3],q[4];
ry(-3.005149904134539) q[4];
ry(1.3616769994097027) q[7];
cx q[4],q[7];
ry(0.05806932416012259) q[4];
ry(-1.5738972382244656) q[7];
cx q[4],q[7];
ry(2.988733854677533) q[5];
ry(-2.529241414480403) q[6];
cx q[5],q[6];
ry(-0.939896823483749) q[5];
ry(2.1329171461013563) q[6];
cx q[5],q[6];
ry(-0.004047080568179595) q[0];
ry(-2.242578201227675) q[1];
cx q[0],q[1];
ry(-2.8350906264133107) q[0];
ry(-2.793173906763435) q[1];
cx q[0],q[1];
ry(-2.663897390002448) q[2];
ry(-1.136748740475742) q[3];
cx q[2],q[3];
ry(3.1328621890415187) q[2];
ry(0.9125620092709292) q[3];
cx q[2],q[3];
ry(-1.5180440157422752) q[4];
ry(0.8660448626700319) q[5];
cx q[4],q[5];
ry(-0.8308635459783561) q[4];
ry(0.323979370375201) q[5];
cx q[4],q[5];
ry(0.5955196392492024) q[6];
ry(1.514670414671767) q[7];
cx q[6],q[7];
ry(-0.6844443160357745) q[6];
ry(2.502895028423214) q[7];
cx q[6],q[7];
ry(-0.6637087315763053) q[0];
ry(2.6119711305384197) q[2];
cx q[0],q[2];
ry(1.2821440843810885) q[0];
ry(-2.570706000380927) q[2];
cx q[0],q[2];
ry(1.1632186141590335) q[2];
ry(0.4239350678609144) q[4];
cx q[2],q[4];
ry(-2.7473371560821516) q[2];
ry(2.6872297426885936) q[4];
cx q[2],q[4];
ry(1.832734275120838) q[4];
ry(-1.6087872646529329) q[6];
cx q[4],q[6];
ry(1.6689164349484191) q[4];
ry(1.4196866373099184) q[6];
cx q[4],q[6];
ry(0.5840556394024077) q[1];
ry(-2.4859562117432703) q[3];
cx q[1],q[3];
ry(3.0067984291350887) q[1];
ry(-2.791119478958456) q[3];
cx q[1],q[3];
ry(-1.7545835331924897) q[3];
ry(-2.71827861047362) q[5];
cx q[3],q[5];
ry(0.4710651275550773) q[3];
ry(-2.167651869206436) q[5];
cx q[3],q[5];
ry(2.0866516426053217) q[5];
ry(-0.6958111540389265) q[7];
cx q[5],q[7];
ry(1.8741550566690854) q[5];
ry(3.0531271282784402) q[7];
cx q[5],q[7];
ry(1.657314796108107) q[0];
ry(-1.5548320151998911) q[3];
cx q[0],q[3];
ry(2.853489090967777) q[0];
ry(1.3065625926689257) q[3];
cx q[0],q[3];
ry(1.5194219316199251) q[1];
ry(2.1268774873827363) q[2];
cx q[1],q[2];
ry(-0.7273031512027198) q[1];
ry(-0.01834367139637691) q[2];
cx q[1],q[2];
ry(-0.4063723710559953) q[2];
ry(0.06799394535241209) q[5];
cx q[2],q[5];
ry(-0.4060035412110965) q[2];
ry(1.8694657395704555) q[5];
cx q[2],q[5];
ry(-0.3527916411630309) q[3];
ry(1.042757622808627) q[4];
cx q[3],q[4];
ry(-1.897770265709065) q[3];
ry(1.4920463383014424) q[4];
cx q[3],q[4];
ry(2.660773193839818) q[4];
ry(-1.269915686349849) q[7];
cx q[4],q[7];
ry(2.6108822181082156) q[4];
ry(1.451833838534066) q[7];
cx q[4],q[7];
ry(1.1643810492497968) q[5];
ry(2.654311702595627) q[6];
cx q[5],q[6];
ry(-2.537915995452381) q[5];
ry(-0.8794064071340403) q[6];
cx q[5],q[6];
ry(-0.22754744419106074) q[0];
ry(-2.8367849887550793) q[1];
cx q[0],q[1];
ry(1.6002971143224582) q[0];
ry(-0.7235371253945498) q[1];
cx q[0],q[1];
ry(-0.5126511664006886) q[2];
ry(0.6548829195953925) q[3];
cx q[2],q[3];
ry(-2.968046077358436) q[2];
ry(-1.9050729267964719) q[3];
cx q[2],q[3];
ry(0.1691024234669337) q[4];
ry(-0.4073336738124272) q[5];
cx q[4],q[5];
ry(-1.3959320006552058) q[4];
ry(-1.0097957028890612) q[5];
cx q[4],q[5];
ry(1.928738159120952) q[6];
ry(3.0463122382928485) q[7];
cx q[6],q[7];
ry(-1.05104379294566) q[6];
ry(-0.6140045458230381) q[7];
cx q[6],q[7];
ry(-1.4101301950278309) q[0];
ry(1.733118899357386) q[2];
cx q[0],q[2];
ry(1.487541624408272) q[0];
ry(-2.895117959643869) q[2];
cx q[0],q[2];
ry(-0.9473271432705771) q[2];
ry(0.48301358406276895) q[4];
cx q[2],q[4];
ry(-0.5775038235543873) q[2];
ry(2.3967320664475906) q[4];
cx q[2],q[4];
ry(-1.906434046602068) q[4];
ry(-0.4082841530064472) q[6];
cx q[4],q[6];
ry(0.4602625436602983) q[4];
ry(1.7704368851383911) q[6];
cx q[4],q[6];
ry(-2.3215768625347497) q[1];
ry(-2.1443295540426375) q[3];
cx q[1],q[3];
ry(-1.967005424628398) q[1];
ry(-1.985885513678199) q[3];
cx q[1],q[3];
ry(-2.9659638207732715) q[3];
ry(2.0862604365905852) q[5];
cx q[3],q[5];
ry(2.8853790630193332) q[3];
ry(-0.9290162878348902) q[5];
cx q[3],q[5];
ry(-1.6768084071289016) q[5];
ry(-1.7746502014652057) q[7];
cx q[5],q[7];
ry(2.8276414533874976) q[5];
ry(-0.38575226469010726) q[7];
cx q[5],q[7];
ry(-2.4466233395710764) q[0];
ry(-0.41952498591764176) q[3];
cx q[0],q[3];
ry(2.8093051565973397) q[0];
ry(-0.882693279157224) q[3];
cx q[0],q[3];
ry(-2.3669123793525557) q[1];
ry(-0.15521547442615474) q[2];
cx q[1],q[2];
ry(-0.6765935341251863) q[1];
ry(1.8136048149345063) q[2];
cx q[1],q[2];
ry(2.2276257675668103) q[2];
ry(2.801637980191263) q[5];
cx q[2],q[5];
ry(1.57998156798085) q[2];
ry(-1.820601252428638) q[5];
cx q[2],q[5];
ry(-1.716029940225434) q[3];
ry(2.8853975461624275) q[4];
cx q[3],q[4];
ry(1.5268780168600067) q[3];
ry(-2.9527208253047146) q[4];
cx q[3],q[4];
ry(0.46645409183819975) q[4];
ry(2.624116572077297) q[7];
cx q[4],q[7];
ry(-2.8988831636669854) q[4];
ry(-0.6611239611937849) q[7];
cx q[4],q[7];
ry(-0.3511439956650504) q[5];
ry(0.3399980068189911) q[6];
cx q[5],q[6];
ry(1.6682164871077412) q[5];
ry(1.0945399173279782) q[6];
cx q[5],q[6];
ry(0.8183429338165851) q[0];
ry(0.4132575688863884) q[1];
cx q[0],q[1];
ry(1.0017345177980217) q[0];
ry(-2.6574656375407604) q[1];
cx q[0],q[1];
ry(0.4750551549038907) q[2];
ry(0.09093189822782832) q[3];
cx q[2],q[3];
ry(-0.9924507143569641) q[2];
ry(-2.4539402810857043) q[3];
cx q[2],q[3];
ry(0.925442643484617) q[4];
ry(1.977420702214505) q[5];
cx q[4],q[5];
ry(0.03607809846766763) q[4];
ry(-1.9398602532600915) q[5];
cx q[4],q[5];
ry(-2.622286159074178) q[6];
ry(2.54620929510015) q[7];
cx q[6],q[7];
ry(1.0184279761419406) q[6];
ry(2.6535671439046626) q[7];
cx q[6],q[7];
ry(2.282255335792012) q[0];
ry(-0.11131785738835731) q[2];
cx q[0],q[2];
ry(2.83830137827818) q[0];
ry(2.674494351889915) q[2];
cx q[0],q[2];
ry(-2.884239794264203) q[2];
ry(-3.096770760100599) q[4];
cx q[2],q[4];
ry(-1.28434763956199) q[2];
ry(-1.2624370864901955) q[4];
cx q[2],q[4];
ry(-1.6295809041918066) q[4];
ry(2.9621496375532477) q[6];
cx q[4],q[6];
ry(-0.4228680059102521) q[4];
ry(1.8115587028546232) q[6];
cx q[4],q[6];
ry(2.3853333429753047) q[1];
ry(2.449719674660952) q[3];
cx q[1],q[3];
ry(-2.0734834155571304) q[1];
ry(2.137322531467409) q[3];
cx q[1],q[3];
ry(-2.122151910166232) q[3];
ry(1.8620126380287854) q[5];
cx q[3],q[5];
ry(2.523259951574788) q[3];
ry(-2.579319724832279) q[5];
cx q[3],q[5];
ry(-1.2674438661540792) q[5];
ry(-1.9797877357991256) q[7];
cx q[5],q[7];
ry(2.085100195306844) q[5];
ry(-0.42511164443707045) q[7];
cx q[5],q[7];
ry(1.1087562792108523) q[0];
ry(3.1034272808812404) q[3];
cx q[0],q[3];
ry(-1.8784265620107032) q[0];
ry(0.48223368657298504) q[3];
cx q[0],q[3];
ry(-2.647391728154272) q[1];
ry(0.8496909350005044) q[2];
cx q[1],q[2];
ry(2.0742251236329485) q[1];
ry(1.8032982240372744) q[2];
cx q[1],q[2];
ry(-2.13744109899733) q[2];
ry(-2.327456453007601) q[5];
cx q[2],q[5];
ry(-0.7210129843362048) q[2];
ry(-2.508602802634707) q[5];
cx q[2],q[5];
ry(-0.6995760664061448) q[3];
ry(0.8753140301960629) q[4];
cx q[3],q[4];
ry(-1.6803854053353173) q[3];
ry(-3.091609069209202) q[4];
cx q[3],q[4];
ry(-1.013879043471654) q[4];
ry(0.052283961788637257) q[7];
cx q[4],q[7];
ry(2.3250281588438524) q[4];
ry(-0.6052150842023539) q[7];
cx q[4],q[7];
ry(-0.6459905324714192) q[5];
ry(-2.1341388203155645) q[6];
cx q[5],q[6];
ry(-1.2756286662801157) q[5];
ry(2.540971735121856) q[6];
cx q[5],q[6];
ry(0.9657529498874267) q[0];
ry(0.8545180381925697) q[1];
cx q[0],q[1];
ry(2.8051487785480176) q[0];
ry(1.7684065351868354) q[1];
cx q[0],q[1];
ry(0.39294254894260905) q[2];
ry(1.1960259369684838) q[3];
cx q[2],q[3];
ry(-1.5691638068574818) q[2];
ry(-1.6735838598138084) q[3];
cx q[2],q[3];
ry(-2.178573070171004) q[4];
ry(-2.0614824075344167) q[5];
cx q[4],q[5];
ry(2.106642725882077) q[4];
ry(-0.7414330089747224) q[5];
cx q[4],q[5];
ry(1.7451993746000165) q[6];
ry(3.079566003108444) q[7];
cx q[6],q[7];
ry(0.06440339162480856) q[6];
ry(-2.58459457978004) q[7];
cx q[6],q[7];
ry(-2.6473102531073525) q[0];
ry(0.13605818098384304) q[2];
cx q[0],q[2];
ry(-0.2862087696037019) q[0];
ry(-0.20216658862891593) q[2];
cx q[0],q[2];
ry(2.2830725556059717) q[2];
ry(-1.4699892901039187) q[4];
cx q[2],q[4];
ry(-2.9993037608570945) q[2];
ry(-1.306566841474319) q[4];
cx q[2],q[4];
ry(1.794162508487599) q[4];
ry(-1.9506941240581868) q[6];
cx q[4],q[6];
ry(-0.16239681519309895) q[4];
ry(2.6243994082918594) q[6];
cx q[4],q[6];
ry(0.3016139632511541) q[1];
ry(-0.678255161475712) q[3];
cx q[1],q[3];
ry(2.3823640614733503) q[1];
ry(-2.0205918850126463) q[3];
cx q[1],q[3];
ry(2.6078382543230854) q[3];
ry(-2.7086449441223626) q[5];
cx q[3],q[5];
ry(1.5065336825774969) q[3];
ry(-1.574661593224328) q[5];
cx q[3],q[5];
ry(-2.0382263098982287) q[5];
ry(-0.13441633687164334) q[7];
cx q[5],q[7];
ry(0.7192046134494136) q[5];
ry(1.8354196318492013) q[7];
cx q[5],q[7];
ry(-1.6933974359268604) q[0];
ry(-2.23845752418185) q[3];
cx q[0],q[3];
ry(1.2508314333305843) q[0];
ry(-0.6022949901587126) q[3];
cx q[0],q[3];
ry(2.5648315924315157) q[1];
ry(-2.5351780423568626) q[2];
cx q[1],q[2];
ry(2.806560963417534) q[1];
ry(0.5392843722591953) q[2];
cx q[1],q[2];
ry(-1.832119882474828) q[2];
ry(0.6892056840681174) q[5];
cx q[2],q[5];
ry(-1.3435839545726704) q[2];
ry(1.8630718831007274) q[5];
cx q[2],q[5];
ry(2.1197617792023036) q[3];
ry(-2.254151828956386) q[4];
cx q[3],q[4];
ry(0.4401938767279763) q[3];
ry(-2.2271051213396293) q[4];
cx q[3],q[4];
ry(1.9359441865666183) q[4];
ry(-3.006475216361708) q[7];
cx q[4],q[7];
ry(-1.5795122313507663) q[4];
ry(-3.0633764470522937) q[7];
cx q[4],q[7];
ry(-1.6890570321822236) q[5];
ry(-2.4537097652607396) q[6];
cx q[5],q[6];
ry(-2.7358981193487324) q[5];
ry(-1.8124455810369786) q[6];
cx q[5],q[6];
ry(1.8815097788177297) q[0];
ry(-1.233573811626795) q[1];
cx q[0],q[1];
ry(1.9621736031652726) q[0];
ry(-3.050828523064461) q[1];
cx q[0],q[1];
ry(0.7754003738579245) q[2];
ry(-2.439446431572888) q[3];
cx q[2],q[3];
ry(-0.1599174570249945) q[2];
ry(2.7077589948422016) q[3];
cx q[2],q[3];
ry(-1.3903930046511084) q[4];
ry(-0.031392183115005245) q[5];
cx q[4],q[5];
ry(-1.7451911528023842) q[4];
ry(0.48848210931206865) q[5];
cx q[4],q[5];
ry(0.25086495702756867) q[6];
ry(-1.5856250133149672) q[7];
cx q[6],q[7];
ry(0.6418336811104978) q[6];
ry(-2.6761439839303507) q[7];
cx q[6],q[7];
ry(-1.8052686096981265) q[0];
ry(0.6892936379060535) q[2];
cx q[0],q[2];
ry(-1.261656687033656) q[0];
ry(1.1186725382173996) q[2];
cx q[0],q[2];
ry(1.8505644152325447) q[2];
ry(-3.0348991660317624) q[4];
cx q[2],q[4];
ry(-1.319023056577767) q[2];
ry(1.0887942325109885) q[4];
cx q[2],q[4];
ry(2.151759234560887) q[4];
ry(-1.586041934098546) q[6];
cx q[4],q[6];
ry(0.7120687542538482) q[4];
ry(-0.669611838879447) q[6];
cx q[4],q[6];
ry(-2.7530229606323786) q[1];
ry(1.7804429806380693) q[3];
cx q[1],q[3];
ry(-2.320483049919571) q[1];
ry(-2.9308805691110296) q[3];
cx q[1],q[3];
ry(-0.563275408057832) q[3];
ry(-0.44911825787813725) q[5];
cx q[3],q[5];
ry(0.35017633707666374) q[3];
ry(-2.569939074021053) q[5];
cx q[3],q[5];
ry(1.3819644400527071) q[5];
ry(-1.2641937304314776) q[7];
cx q[5],q[7];
ry(-1.913864444307087) q[5];
ry(0.9696580970946793) q[7];
cx q[5],q[7];
ry(-2.9897406757002423) q[0];
ry(-2.0901858249458645) q[3];
cx q[0],q[3];
ry(1.8437201170236168) q[0];
ry(-0.8460384595600488) q[3];
cx q[0],q[3];
ry(2.75898201354366) q[1];
ry(2.472316762251206) q[2];
cx q[1],q[2];
ry(1.261146836029913) q[1];
ry(1.4896308688052047) q[2];
cx q[1],q[2];
ry(-1.8577524330692317) q[2];
ry(2.29332283131161) q[5];
cx q[2],q[5];
ry(-0.8203605827241462) q[2];
ry(0.2557611796640749) q[5];
cx q[2],q[5];
ry(-0.6719115145575597) q[3];
ry(-0.2461440337563907) q[4];
cx q[3],q[4];
ry(2.217834383006613) q[3];
ry(-2.222965435308393) q[4];
cx q[3],q[4];
ry(2.953927014071968) q[4];
ry(1.6577803166884242) q[7];
cx q[4],q[7];
ry(1.0958225244203161) q[4];
ry(2.253208225874746) q[7];
cx q[4],q[7];
ry(-2.3272758117021226) q[5];
ry(-2.425429888402787) q[6];
cx q[5],q[6];
ry(-1.8396854706703456) q[5];
ry(-2.546810506426719) q[6];
cx q[5],q[6];
ry(-2.8804792210686254) q[0];
ry(-2.8316412902686303) q[1];
ry(-3.0234450381554194) q[2];
ry(1.7405733435979265) q[3];
ry(0.04595453511232856) q[4];
ry(1.001655078725105) q[5];
ry(1.6822952162879428) q[6];
ry(0.6063376930088589) q[7];