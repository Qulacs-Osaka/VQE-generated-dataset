OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.5620462678088103) q[0];
rz(0.37605986893149806) q[0];
ry(0.23163318702160218) q[1];
rz(-1.6470043159415342) q[1];
ry(-2.0849609946480303) q[2];
rz(2.9945391420392675) q[2];
ry(-2.0901599910425883) q[3];
rz(2.1645768188464105) q[3];
ry(-2.858829644681368) q[4];
rz(2.4240305752462867) q[4];
ry(-3.0341017088463222) q[5];
rz(-2.9967188107300187) q[5];
ry(-0.3694973395595715) q[6];
rz(-0.7308653873664781) q[6];
ry(-0.0261048583538809) q[7];
rz(-0.7935023873715573) q[7];
ry(2.3407058099608715) q[8];
rz(-1.2803954546580918) q[8];
ry(3.095952051975398) q[9];
rz(-2.1060643913397863) q[9];
ry(-2.191172230598536) q[10];
rz(-0.5442565773828744) q[10];
ry(2.8053253754365812) q[11];
rz(1.9710162830683018) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.2776103100802434) q[0];
rz(-2.7624614547687587) q[0];
ry(0.20337691502566438) q[1];
rz(0.2808792857765508) q[1];
ry(-1.2276316843348272) q[2];
rz(-0.020128115554957304) q[2];
ry(-1.424192005211517) q[3];
rz(0.4981232566220344) q[3];
ry(-2.8001922627638915) q[4];
rz(0.8225662300862595) q[4];
ry(1.3658252736243208) q[5];
rz(-1.3081362066313473) q[5];
ry(1.3050375078905194) q[6];
rz(-1.1589256519087048) q[6];
ry(2.2685994422810225) q[7];
rz(-0.5727913472570956) q[7];
ry(-2.3144005915631496) q[8];
rz(-0.49187442783304736) q[8];
ry(-1.2607811172167624) q[9];
rz(0.8584855608515469) q[9];
ry(1.4560976435364326) q[10];
rz(2.1745575201411036) q[10];
ry(2.699114770307589) q[11];
rz(-0.3331013574250346) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.8057984347418765) q[0];
rz(0.8654605167777295) q[0];
ry(-1.8941839376991059) q[1];
rz(2.866380252500436) q[1];
ry(0.20644978134861083) q[2];
rz(1.992260229778979) q[2];
ry(-1.8862741866555828) q[3];
rz(-1.0738919713442032) q[3];
ry(3.011281164076592) q[4];
rz(0.15314843748007242) q[4];
ry(1.274305685845182) q[5];
rz(2.9944744844267612) q[5];
ry(2.8855567375687454) q[6];
rz(2.746657876783148) q[6];
ry(-0.0059743343079254885) q[7];
rz(-2.735769640977207) q[7];
ry(0.0917914434377057) q[8];
rz(-0.4872544424689645) q[8];
ry(-3.051881070100656) q[9];
rz(-2.5820162958122523) q[9];
ry(-3.034271955577764) q[10];
rz(0.19069714001982874) q[10];
ry(-0.971393508288317) q[11];
rz(2.2513719485332824) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.056796960452440454) q[0];
rz(-1.7027319927617075) q[0];
ry(1.0724632000392798) q[1];
rz(-1.8189553161050798) q[1];
ry(2.0636467892532853) q[2];
rz(-0.6509482507798129) q[2];
ry(-2.940537166703239) q[3];
rz(0.0600300889780927) q[3];
ry(-1.634415802290754) q[4];
rz(-2.4219420343831697) q[4];
ry(-2.8733635827066104) q[5];
rz(-0.019335492357846203) q[5];
ry(-0.28788708206590224) q[6];
rz(1.431790315478894) q[6];
ry(-0.4452050416914446) q[7];
rz(-1.745768853262724) q[7];
ry(0.2116979595350088) q[8];
rz(0.0517536309623503) q[8];
ry(1.3338704726843944) q[9];
rz(0.8587729205179744) q[9];
ry(0.11931809983057153) q[10];
rz(2.286307580719458) q[10];
ry(0.6557747533428069) q[11];
rz(0.6882513125627554) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.0333086565430123) q[0];
rz(-2.7795947975401445) q[0];
ry(-0.4174970785609595) q[1];
rz(1.10220656278983) q[1];
ry(2.351359245997196) q[2];
rz(1.4067553796775805) q[2];
ry(1.516668385225918) q[3];
rz(-0.6777510180105906) q[3];
ry(3.0162815235114033) q[4];
rz(1.6919645799057843) q[4];
ry(2.9185737611290934) q[5];
rz(-3.0067958542271507) q[5];
ry(-3.0927373917607985) q[6];
rz(2.0020816934125305) q[6];
ry(-3.1347631201189103) q[7];
rz(3.0127015575180263) q[7];
ry(0.1339814654151299) q[8];
rz(0.31186796999997135) q[8];
ry(-3.083314584406151) q[9];
rz(-0.3167989914686702) q[9];
ry(2.628273566946084) q[10];
rz(-0.4707564217091313) q[10];
ry(0.6266597305244916) q[11];
rz(-2.8602172991305403) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.1664989549290345) q[0];
rz(-2.1120763531643965) q[0];
ry(-1.234037066971438) q[1];
rz(-3.09208729722451) q[1];
ry(-1.9870856879087204) q[2];
rz(-2.323139185798808) q[2];
ry(1.056085385418199) q[3];
rz(1.4735709754410973) q[3];
ry(-0.5208030001418624) q[4];
rz(2.9806667999499754) q[4];
ry(-1.8729951888190657) q[5];
rz(-0.44785982406614466) q[5];
ry(0.17345131988780468) q[6];
rz(-0.7346976531561804) q[6];
ry(2.201179841186921) q[7];
rz(2.6140368467715356) q[7];
ry(0.3864354118913437) q[8];
rz(-2.9089535019371673) q[8];
ry(3.072016865147118) q[9];
rz(1.145485495050409) q[9];
ry(-2.3251464952629584) q[10];
rz(-2.2484473346186267) q[10];
ry(-1.6664607634099748) q[11];
rz(1.353493287048508) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.1153228216375428) q[0];
rz(1.185250341028358) q[0];
ry(2.3124444932654526) q[1];
rz(-1.0779872868707367) q[1];
ry(1.8760411319401031) q[2];
rz(2.877897294365542) q[2];
ry(1.004780434029955) q[3];
rz(1.971851378944957) q[3];
ry(3.0938391279738386) q[4];
rz(2.1344410802337794) q[4];
ry(-3.110506367751285) q[5];
rz(-2.5929763983056837) q[5];
ry(2.82816392759781) q[6];
rz(-2.3893453721689437) q[6];
ry(3.1292014501620726) q[7];
rz(-2.1325954509533442) q[7];
ry(-1.0033559273087498) q[8];
rz(1.9960646181699948) q[8];
ry(0.15717147277173427) q[9];
rz(2.6316986775989046) q[9];
ry(-0.7058210764355808) q[10];
rz(1.5563022059063916) q[10];
ry(0.24823029559364823) q[11];
rz(-1.0450184778327263) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.1401593416019367) q[0];
rz(1.1663502551215141) q[0];
ry(0.5942863831290534) q[1];
rz(2.9800923785177225) q[1];
ry(2.651395395127672) q[2];
rz(2.640984030502813) q[2];
ry(0.24789138840621483) q[3];
rz(-0.3179775969484596) q[3];
ry(-1.89415948678365) q[4];
rz(1.0431450265252487) q[4];
ry(-2.8919919882410126) q[5];
rz(-0.19324407274559993) q[5];
ry(-0.7362162717923048) q[6];
rz(2.334800435340076) q[6];
ry(1.0575653151749909) q[7];
rz(-2.0308153693760245) q[7];
ry(0.6303155787281195) q[8];
rz(-0.43191307607203155) q[8];
ry(-2.495504959092315) q[9];
rz(-0.08454229299653893) q[9];
ry(2.922247050787697) q[10];
rz(1.43174354663595) q[10];
ry(2.8460219067761505) q[11];
rz(1.0097916543241947) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.3338915736219077) q[0];
rz(0.2648494410989599) q[0];
ry(-0.7289142436900624) q[1];
rz(-1.7541381732374643) q[1];
ry(0.11836459615558326) q[2];
rz(-1.9593135803806687) q[2];
ry(-1.5619199852292986) q[3];
rz(1.4041985303890643) q[3];
ry(0.03267622249340206) q[4];
rz(0.6231054618287067) q[4];
ry(-0.04793234574545744) q[5];
rz(-0.5929335499883832) q[5];
ry(3.1382005485573727) q[6];
rz(-0.5714399451159358) q[6];
ry(0.0017389289364064808) q[7];
rz(0.6059569242783178) q[7];
ry(-1.0390702736866086) q[8];
rz(-0.18429357448746167) q[8];
ry(-0.02702973715350776) q[9];
rz(-0.9264076064912639) q[9];
ry(3.1386274215258396) q[10];
rz(1.1278087648268391) q[10];
ry(2.8647312401041827) q[11];
rz(0.11814973778966201) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.9916090399400592) q[0];
rz(-0.959304872463167) q[0];
ry(1.5539765553737386) q[1];
rz(-2.8964477750710103) q[1];
ry(-2.175156848719549) q[2];
rz(0.5111742335661577) q[2];
ry(2.265496475497808) q[3];
rz(-0.10069145677720746) q[3];
ry(-2.6763682851012147) q[4];
rz(0.12848629358923205) q[4];
ry(-2.9768181944069627) q[5];
rz(-0.3666783800483602) q[5];
ry(-1.132708719695485) q[6];
rz(3.116624459454539) q[6];
ry(-1.9338434290437707) q[7];
rz(0.770592430808926) q[7];
ry(-1.4237842627726334) q[8];
rz(2.76396216829839) q[8];
ry(0.7324411998289363) q[9];
rz(0.3721814421203913) q[9];
ry(-2.608146954066657) q[10];
rz(-2.7096142858950865) q[10];
ry(-1.8559492794138333) q[11];
rz(0.7131137714255119) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.332566586328415) q[0];
rz(-1.3740663929395514) q[0];
ry(1.0599609756073578) q[1];
rz(2.1128743158020202) q[1];
ry(0.6363217346732399) q[2];
rz(0.4218810717685437) q[2];
ry(-1.297796128278886) q[3];
rz(-2.046004097436694) q[3];
ry(-0.08380305011749822) q[4];
rz(2.441863350968271) q[4];
ry(0.008658386943792175) q[5];
rz(-2.420374103884437) q[5];
ry(-0.005147072997616526) q[6];
rz(0.08142695012886758) q[6];
ry(3.133784948721346) q[7];
rz(-0.05604336722414427) q[7];
ry(-3.014590525026459) q[8];
rz(-1.960485051793814) q[8];
ry(-2.7791249919238536) q[9];
rz(-0.40330722083861226) q[9];
ry(-1.8845738515996675) q[10];
rz(0.19145902850610153) q[10];
ry(1.8835191817483363) q[11];
rz(-0.9374395174744237) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.018415933546673) q[0];
rz(-1.840319779441219) q[0];
ry(0.838524602359186) q[1];
rz(-1.6228868287847968) q[1];
ry(-2.8951907427964954) q[2];
rz(-2.049637045668754) q[2];
ry(1.5846307812696696) q[3];
rz(0.33492438185362167) q[3];
ry(2.6498062426780957) q[4];
rz(-2.158160956445802) q[4];
ry(3.098147478620608) q[5];
rz(2.119036516356805) q[5];
ry(0.16633465808819062) q[6];
rz(0.1392612817275993) q[6];
ry(1.7102626280501614) q[7];
rz(0.8425672070622313) q[7];
ry(-2.895318897721762) q[8];
rz(-1.3527620536704057) q[8];
ry(-0.26725892218131175) q[9];
rz(0.3884542251556258) q[9];
ry(0.43236492053479386) q[10];
rz(-2.6114532264010832) q[10];
ry(-0.22010297047629432) q[11];
rz(-2.2633872845485925) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.110513187535426) q[0];
rz(-2.787952294028205) q[0];
ry(-1.7285162990068843) q[1];
rz(0.1231199719610492) q[1];
ry(1.9900133360234558) q[2];
rz(2.110445449155131) q[2];
ry(1.3238021645280575) q[3];
rz(-0.873140720520042) q[3];
ry(-0.03780509361104417) q[4];
rz(1.192363915416805) q[4];
ry(-1.1712378374645362) q[5];
rz(0.5887675798872705) q[5];
ry(-0.007946495591270235) q[6];
rz(0.37154326842547825) q[6];
ry(0.0006640918205662863) q[7];
rz(0.20451574647649906) q[7];
ry(1.86307256329997) q[8];
rz(-2.988567031047752) q[8];
ry(-0.2544402902705414) q[9];
rz(0.5899731378643294) q[9];
ry(-0.4806492867485423) q[10];
rz(-2.9922499284232926) q[10];
ry(0.28461489625228076) q[11];
rz(-0.54372287330144) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.8102051828185884) q[0];
rz(1.737415981648769) q[0];
ry(-2.487587920218214) q[1];
rz(-0.4425236167154485) q[1];
ry(1.4068867938484484) q[2];
rz(0.9456964199810267) q[2];
ry(1.0737796846482697) q[3];
rz(0.10886614880725415) q[3];
ry(-2.845667380494191) q[4];
rz(0.5584421649065234) q[4];
ry(0.21813615388410112) q[5];
rz(-0.8937598021545303) q[5];
ry(3.0118991438962928) q[6];
rz(-1.8943184496354846) q[6];
ry(-2.8267652625086686) q[7];
rz(2.098795008496258) q[7];
ry(-3.0005942211425607) q[8];
rz(2.107675006132536) q[8];
ry(2.7018531711382274) q[9];
rz(0.9541582793815737) q[9];
ry(2.654792377204725) q[10];
rz(-2.880155823718011) q[10];
ry(-0.6876524931643839) q[11];
rz(0.19789068623457062) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.7743003270451813) q[0];
rz(1.6447952121287552) q[0];
ry(0.779432305288949) q[1];
rz(-0.5299890196463705) q[1];
ry(2.1598421137045456) q[2];
rz(-0.32103836381328593) q[2];
ry(-1.778109863481466) q[3];
rz(-2.716203428005779) q[3];
ry(-3.0975813427003973) q[4];
rz(-2.690699160183585) q[4];
ry(2.9982883485628284) q[5];
rz(0.8946988243672505) q[5];
ry(-3.130977265134296) q[6];
rz(-1.8884751238984911) q[6];
ry(3.1411699718545867) q[7];
rz(-1.6358070287243134) q[7];
ry(-0.4031000639564412) q[8];
rz(-2.5203456837380056) q[8];
ry(-2.6031498367467343) q[9];
rz(-2.3473414344900365) q[9];
ry(1.063222529413129) q[10];
rz(-1.8461998251698943) q[10];
ry(-1.8983672221475907) q[11];
rz(-1.0525886451222801) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5943404291782215) q[0];
rz(-0.4342834393324635) q[0];
ry(1.5713104711330355) q[1];
rz(2.08467988238002) q[1];
ry(1.5834779898785944) q[2];
rz(-1.951969528638367) q[2];
ry(1.39404627849287) q[3];
rz(2.774613219460235) q[3];
ry(-0.1509914064198741) q[4];
rz(0.45979926189410864) q[4];
ry(0.10507151574107843) q[5];
rz(-2.0288992405436517) q[5];
ry(1.2601589502634) q[6];
rz(-2.688551266363142) q[6];
ry(0.8225546769399694) q[7];
rz(0.6371802093309158) q[7];
ry(2.525699645026553) q[8];
rz(0.9172420501724473) q[8];
ry(0.09375293968216318) q[9];
rz(0.1700673361668777) q[9];
ry(-2.9628923191412904) q[10];
rz(2.644105316592657) q[10];
ry(-1.7697051757177125) q[11];
rz(0.8872350255174004) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.3356237255617245) q[0];
rz(-2.8497430745637815) q[0];
ry(-0.8419926280361293) q[1];
rz(1.7195283435027455) q[1];
ry(1.5480468749439846) q[2];
rz(-1.0139537405504908) q[2];
ry(1.78292894131991) q[3];
rz(-1.6523285896488746) q[3];
ry(-3.09703521646576) q[4];
rz(0.10750059084311481) q[4];
ry(2.974550102184865) q[5];
rz(-2.989104413007967) q[5];
ry(1.1041576345253596) q[6];
rz(-2.769551467087405) q[6];
ry(1.5656404611422525) q[7];
rz(-1.8787258195562695) q[7];
ry(-2.006558825471763) q[8];
rz(-2.157468147890812) q[8];
ry(2.3714518851381166) q[9];
rz(1.5938874635817775) q[9];
ry(-1.550170993270083) q[10];
rz(0.43414233473311103) q[10];
ry(1.2246119189939533) q[11];
rz(-0.3762071977521577) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.3732184295318719) q[0];
rz(1.3505357138810714) q[0];
ry(-0.7124618710106247) q[1];
rz(3.1095189625511814) q[1];
ry(2.108648413806777) q[2];
rz(-0.8378560083566297) q[2];
ry(-1.5471471989201246) q[3];
rz(-1.6735004688258623) q[3];
ry(2.9086793960586754) q[4];
rz(0.2535404360534865) q[4];
ry(0.000728419197424235) q[5];
rz(1.3530757038316656) q[5];
ry(2.4638268968633126) q[6];
rz(-3.022742379239657) q[6];
ry(0.002008305256207892) q[7];
rz(-2.0404833026810922) q[7];
ry(-1.5653796141990737) q[8];
rz(-2.0799148809664807) q[8];
ry(0.15887813551148344) q[9];
rz(0.8279624479993436) q[9];
ry(-1.5309996101496544) q[10];
rz(-1.285867778805436) q[10];
ry(2.1032844180526244) q[11];
rz(2.495518876408584) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.9730366498840501) q[0];
rz(0.5884318009877767) q[0];
ry(2.068727500463391) q[1];
rz(1.7228133542813568) q[1];
ry(-0.4759578874807673) q[2];
rz(-1.759912357275587) q[2];
ry(3.0748593223937175) q[3];
rz(1.1647367865822265) q[3];
ry(-0.8855243369290623) q[4];
rz(0.8617636785505437) q[4];
ry(3.140707381337419) q[5];
rz(0.11563412365123148) q[5];
ry(1.6056295741121747) q[6];
rz(-0.3549376705707526) q[6];
ry(3.1164689871018894) q[7];
rz(-2.514691250152015) q[7];
ry(-2.8944698265847637) q[8];
rz(1.3705064524680628) q[8];
ry(-1.5621788803817989) q[9];
rz(-2.2692159737371638) q[9];
ry(1.673213914733637) q[10];
rz(-1.2364941454217782) q[10];
ry(-0.1001668665915775) q[11];
rz(-2.375967158310635) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.7976827393855355) q[0];
rz(0.04850908819648235) q[0];
ry(-1.5117963887477628) q[1];
rz(-2.0272414465497257) q[1];
ry(-1.8702680186624474) q[2];
rz(-0.33076063827149227) q[2];
ry(-3.059579140688423) q[3];
rz(-1.3651614583159706) q[3];
ry(-0.009530948066708511) q[4];
rz(-1.8226794453529738) q[4];
ry(-3.1331760618200604) q[5];
rz(3.0045257479537057) q[5];
ry(2.582069394269292) q[6];
rz(1.5347433707342553) q[6];
ry(-3.1339131896191414) q[7];
rz(1.7645256108378824) q[7];
ry(-0.008596802880709298) q[8];
rz(-3.0218627704934704) q[8];
ry(3.1415401624206285) q[9];
rz(1.1494097392560612) q[9];
ry(1.6092732336318685) q[10];
rz(3.0609652769301454) q[10];
ry(-0.9462039913204475) q[11];
rz(0.9936191952879899) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.8504665370754614) q[0];
rz(-2.1463106820339233) q[0];
ry(-1.7516365170793418) q[1];
rz(2.2859890455846568) q[1];
ry(2.3634964555521867) q[2];
rz(2.0248151525971343) q[2];
ry(-2.2529563202752234) q[3];
rz(-2.5360644142049975) q[3];
ry(2.7777758534017054) q[4];
rz(-2.536167380127374) q[4];
ry(-0.11269936023366967) q[5];
rz(-1.608097492318926) q[5];
ry(0.7311103632346728) q[6];
rz(0.4854327636172542) q[6];
ry(-0.04546550453883298) q[7];
rz(-2.1336325007242136) q[7];
ry(2.73732026709305) q[8];
rz(1.4496567330698786) q[8];
ry(3.1300831701738563) q[9];
rz(2.59693990360494) q[9];
ry(0.22696740505179136) q[10];
rz(1.9171546526569183) q[10];
ry(-0.010610725957344313) q[11];
rz(3.1194347177391513) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.5629693067960809) q[0];
rz(-2.3438877958478046) q[0];
ry(2.1426852170385464) q[1];
rz(-2.5108394989008835) q[1];
ry(-0.8773217515754375) q[2];
rz(0.4745765309024402) q[2];
ry(3.126311794222312) q[3];
rz(-0.5368217457034402) q[3];
ry(0.0020221864712759896) q[4];
rz(-2.2550350244421646) q[4];
ry(-0.003521578341151077) q[5];
rz(-2.6807697647310853) q[5];
ry(-1.3309206711728832) q[6];
rz(-2.5388053400259456) q[6];
ry(-3.1342520961436025) q[7];
rz(-1.778586192443707) q[7];
ry(-0.003170031986563515) q[8];
rz(2.2001286935745847) q[8];
ry(0.15584428045440007) q[9];
rz(-0.7552233851829141) q[9];
ry(0.006464002440386807) q[10];
rz(-1.8628633568586324) q[10];
ry(0.8668332871970872) q[11];
rz(2.3041480671299825) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.12347322570636063) q[0];
rz(2.0395913369859615) q[0];
ry(-0.23770686982221131) q[1];
rz(0.027399465502816394) q[1];
ry(-1.7385330172484084) q[2];
rz(1.48743135240085) q[2];
ry(-0.6180531854910474) q[3];
rz(2.315954854395571) q[3];
ry(2.2426968658727056) q[4];
rz(-1.967284455218432) q[4];
ry(3.0571762688790214) q[5];
rz(1.0063427252682504) q[5];
ry(-1.1854747724450647) q[6];
rz(-3.0375669238312573) q[6];
ry(1.5971144183218986) q[7];
rz(-1.5125570832980806) q[7];
ry(-2.2110695083211813) q[8];
rz(0.031548590951726574) q[8];
ry(-1.5520306800518124) q[9];
rz(-3.1051035049472384) q[9];
ry(2.7340782990479777) q[10];
rz(-0.028872462734377713) q[10];
ry(0.009617661287831916) q[11];
rz(2.0657610650407174) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.9064607308893963) q[0];
rz(-2.0070474084316885) q[0];
ry(-1.7396456666923619) q[1];
rz(-3.075191024002867) q[1];
ry(2.6890708923387976) q[2];
rz(0.5960603617707827) q[2];
ry(-0.009805302300703644) q[3];
rz(-2.1216154941198866) q[3];
ry(3.1385614932926043) q[4];
rz(1.543637988960531) q[4];
ry(-3.140328294690532) q[5];
rz(-3.0320940817210857) q[5];
ry(3.128013716114446) q[6];
rz(0.3406209156866684) q[6];
ry(0.0010745156518411972) q[7];
rz(1.561884679845964) q[7];
ry(0.3936716646428176) q[8];
rz(-0.00038076266739869044) q[8];
ry(1.634542125260974) q[9];
rz(0.7977856445772371) q[9];
ry(-1.0710198109586964) q[10];
rz(3.0666585926286656) q[10];
ry(-2.9658290645524312) q[11];
rz(0.6308522722320385) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.4173360811857547) q[0];
rz(0.9058855901422183) q[0];
ry(2.9096474256935974) q[1];
rz(-1.7714896451405393) q[1];
ry(1.421601649537241) q[2];
rz(-1.0894525780923878) q[2];
ry(-1.3679757855014187) q[3];
rz(-2.659213384031832) q[3];
ry(1.4322053834628152) q[4];
rz(-0.7978906800197422) q[4];
ry(-1.6166745225795145) q[5];
rz(-0.08940148480097766) q[5];
ry(1.5590281942693318) q[6];
rz(-2.5914191822514816) q[6];
ry(-3.113471222570108) q[7];
rz(-1.1953224854842655) q[7];
ry(1.6476404452315965) q[8];
rz(-3.1185955161771313) q[8];
ry(1.911275841359207) q[9];
rz(1.5451558639929344) q[9];
ry(-1.5940122257746305) q[10];
rz(1.87427241895867) q[10];
ry(-1.6323719295372774) q[11];
rz(0.5521950971841205) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.7214578631961377) q[0];
rz(1.1232371889547799) q[0];
ry(2.784078516605487) q[1];
rz(0.8514761810198652) q[1];
ry(0.5701529344462762) q[2];
rz(-1.807455304480545) q[2];
ry(-2.9500490384004485) q[3];
rz(-0.24721832479873043) q[3];
ry(-0.5138680285072983) q[4];
rz(2.15903511526826) q[4];
ry(0.8038116618113142) q[5];
rz(-3.100652614567255) q[5];
ry(-1.5458553917942914) q[6];
rz(2.3781797672915723) q[6];
ry(-0.019770989749616892) q[7];
rz(-1.890341605678267) q[7];
ry(-3.0968150490647184) q[8];
rz(-2.4066204015079515) q[8];
ry(-2.922790764626867) q[9];
rz(2.2684643792281474) q[9];
ry(-3.1386210942068495) q[10];
rz(0.1687346169275363) q[10];
ry(-0.01119853077191646) q[11];
rz(-2.3131573707678896) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.299271725840093) q[0];
rz(-2.4860001838584482) q[0];
ry(-2.1599923839105184) q[1];
rz(-2.0440417764183483) q[1];
ry(0.14051935147329522) q[2];
rz(-2.3280251938070964) q[2];
ry(2.861889604100554) q[3];
rz(-0.00023809918205053296) q[3];
ry(-3.137024327609778) q[4];
rz(-2.4916189809701983) q[4];
ry(-0.05292596742554423) q[5];
rz(-3.1132275757756127) q[5];
ry(3.1354809873882794) q[6];
rz(2.3638468417986656) q[6];
ry(2.7228288850445757) q[7];
rz(-3.136941351031198) q[7];
ry(-3.1350181551704037) q[8];
rz(0.7132803943357455) q[8];
ry(0.8680275077766432) q[9];
rz(-1.1440937066275894) q[9];
ry(0.07894663688615208) q[10];
rz(1.6777253970395805) q[10];
ry(0.29562032183889214) q[11];
rz(0.2622234431405807) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.7788889484763744) q[0];
rz(2.9907015409902757) q[0];
ry(1.2698317572533435) q[1];
rz(-1.853237611860762) q[1];
ry(2.6762188137211527) q[2];
rz(1.5534810049353196) q[2];
ry(1.6961239694689931) q[3];
rz(-3.119424462635562) q[3];
ry(-1.5649294076823477) q[4];
rz(-1.2104556952727172) q[4];
ry(-2.3306058606372764) q[5];
rz(0.05367522732955976) q[5];
ry(-0.02080625098387137) q[6];
rz(-3.1254546369617198) q[6];
ry(-1.5482909437117849) q[7];
rz(-1.8477373550471663) q[7];
ry(1.8207353484448958) q[8];
rz(0.0009887602303279063) q[8];
ry(1.5733609787951255) q[9];
rz(-2.8434066589771585) q[9];
ry(-1.602767311112663) q[10];
rz(3.1410875804907388) q[10];
ry(0.2231088273798747) q[11];
rz(-1.965612954437442) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.9993423109562628) q[0];
rz(1.5328971018428321) q[0];
ry(-2.8833921188970884) q[1];
rz(-1.5051735483890576) q[1];
ry(1.4894171085923773) q[2];
rz(3.1046022564663245) q[2];
ry(0.6256689815557669) q[3];
rz(1.7739221814141448) q[3];
ry(-3.141084357572993) q[4];
rz(1.4335052758124132) q[4];
ry(-0.15494018282679534) q[5];
rz(-0.005332426138460348) q[5];
ry(-0.16789525161810204) q[6];
rz(-2.2535411671863486) q[6];
ry(0.002094255089057006) q[7];
rz(0.36115752317838457) q[7];
ry(-0.6625346578899469) q[8];
rz(0.0678474058994194) q[8];
ry(-1.5879561221836755) q[9];
rz(-0.0017722860727440448) q[9];
ry(-1.9054055046828802) q[10];
rz(0.013753039645201191) q[10];
ry(3.125308426499826) q[11];
rz(-0.06488849902066107) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.6485796570766451) q[0];
rz(-1.0632383731689752) q[0];
ry(1.6274043501556248) q[1];
rz(-1.6509338566887255) q[1];
ry(-1.5692703300532864) q[2];
rz(-0.044546187041186514) q[2];
ry(0.0059265828743376835) q[3];
rz(2.6489822846001583) q[3];
ry(-1.599916533315489) q[4];
rz(0.31479922834765806) q[4];
ry(1.616472822142419) q[5];
rz(-0.6961539768562996) q[5];
ry(0.14187296043234188) q[6];
rz(2.704129012175743) q[6];
ry(-2.446188681376652) q[7];
rz(-1.7604221843563423) q[7];
ry(1.5762527650125457) q[8];
rz(2.407511873471486) q[8];
ry(-1.5597578901969729) q[9];
rz(-0.42466655462880887) q[9];
ry(-1.3075856541952728) q[10];
rz(-1.563196891193769) q[10];
ry(-0.007416289011627164) q[11];
rz(-0.7897328124743207) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5477113134535148) q[0];
rz(0.02067061653749747) q[0];
ry(-2.915378101326609) q[1];
rz(1.4986993471600263) q[1];
ry(3.07416344642078) q[2];
rz(2.1682113111246704) q[2];
ry(1.061531975159558) q[3];
rz(-2.1165789983562924) q[3];
ry(-2.1175193788027533) q[4];
rz(-2.0489846948113284) q[4];
ry(-0.004894829610620714) q[5];
rz(0.2891784209842749) q[5];
ry(-3.1415011231213) q[6];
rz(0.4524253407886443) q[6];
ry(-0.0012456361456107958) q[7];
rz(0.2334993115268098) q[7];
ry(-0.00019391595194484523) q[8];
rz(-0.2182941896712167) q[8];
ry(3.1320835750462743) q[9];
rz(2.370803246042711) q[9];
ry(-2.4457384053261055) q[10];
rz(-0.6810438757765583) q[10];
ry(-0.2252677883179679) q[11];
rz(0.7557971171714392) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5925277861188107) q[0];
rz(-1.5446035272473069) q[0];
ry(1.5294596938833842) q[1];
rz(1.5417397540574154) q[1];
ry(-0.0013829506137122607) q[2];
rz(-0.634670931976081) q[2];
ry(-0.0003685079345912508) q[3];
rz(2.657112394352651) q[3];
ry(-3.140433918898753) q[4];
rz(2.609651365849468) q[4];
ry(-0.004195396287069997) q[5];
rz(-1.161782957625825) q[5];
ry(1.6108022212652555) q[6];
rz(1.6836005587705252) q[6];
ry(1.623236709099495) q[7];
rz(0.8751900682723567) q[7];
ry(-3.0585142968448875) q[8];
rz(0.6135821182721847) q[8];
ry(-0.008368380460582081) q[9];
rz(1.9143675394683313) q[9];
ry(0.026602866918043786) q[10];
rz(-2.4494687632355725) q[10];
ry(3.1374694549755673) q[11];
rz(0.7721148778794956) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.9799515431371546) q[0];
rz(2.8525765024007117) q[0];
ry(1.5729251050739803) q[1];
rz(-0.5034261333991843) q[1];
ry(1.5731391551579712) q[2];
rz(-0.3517286320216165) q[2];
ry(2.5580681697978833) q[3];
rz(1.573145938901221) q[3];
ry(1.6604860805395356) q[4];
rz(2.306914286698012) q[4];
ry(1.5716970441463027) q[5];
rz(-2.127724147332527) q[5];
ry(-1.5694218350489084) q[6];
rz(-0.2936241013814156) q[6];
ry(-1.570745221458146) q[7];
rz(2.6118037127543943) q[7];
ry(1.572349382447357) q[8];
rz(-0.292381988817489) q[8];
ry(1.5767160495688506) q[9];
rz(-1.2529099743633045) q[9];
ry(-2.2664547018564214) q[10];
rz(-1.8530139654228472) q[10];
ry(0.22907397415244102) q[11];
rz(2.077339865018673) q[11];