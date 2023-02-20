OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.7902403785813386) q[0];
rz(-1.501251796795492) q[0];
ry(1.7467564426598678) q[1];
rz(-0.3956820384726898) q[1];
ry(1.7282686883467002) q[2];
rz(0.6426507220330073) q[2];
ry(2.0090397631314216) q[3];
rz(1.88838924582624) q[3];
ry(3.1410481410428677) q[4];
rz(-0.23411188611140066) q[4];
ry(1.9585123442110512) q[5];
rz(2.1464616096605083) q[5];
ry(2.9914039170067954) q[6];
rz(-0.9331539286168393) q[6];
ry(-3.140751988258357) q[7];
rz(-2.781723345799287) q[7];
ry(2.1882971338641894) q[8];
rz(-3.0762467480065783) q[8];
ry(-2.449793139678229) q[9];
rz(-2.400812258298584) q[9];
ry(-1.8169350358047796) q[10];
rz(2.457518750849401) q[10];
ry(0.6689821536548202) q[11];
rz(-0.5751818580027712) q[11];
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
ry(1.6627787852717404) q[0];
rz(-1.5059952430517072) q[0];
ry(-2.8420801095431716) q[1];
rz(-1.1717956418950262) q[1];
ry(0.7619393836045614) q[2];
rz(2.81090123943675) q[2];
ry(0.43285581674496854) q[3];
rz(-0.4714507772604995) q[3];
ry(-3.1209199221985053) q[4];
rz(1.0492099812899933) q[4];
ry(2.1269518876593185) q[5];
rz(-0.14203373690916582) q[5];
ry(0.034440616531288264) q[6];
rz(2.804136009352292) q[6];
ry(2.8709302222457396) q[7];
rz(-0.8804414420842689) q[7];
ry(1.415580355718241) q[8];
rz(2.936280630901308) q[8];
ry(-1.9021865331540706) q[9];
rz(1.5613970496378087) q[9];
ry(2.2713838618331397) q[10];
rz(0.03881933449836269) q[10];
ry(1.5512504708613066) q[11];
rz(1.3318158086981322) q[11];
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
ry(-0.0782187106431381) q[0];
rz(0.08889793884796494) q[0];
ry(1.5783771039988224) q[1];
rz(-1.091836747351877) q[1];
ry(2.412741672042071) q[2];
rz(-2.1006799333839137) q[2];
ry(-2.093765843215203) q[3];
rz(2.986838236629524) q[3];
ry(-0.12068945039758905) q[4];
rz(-1.6958704771903754) q[4];
ry(0.017326609915725658) q[5];
rz(-0.9901273194875331) q[5];
ry(2.6342641528497404) q[6];
rz(-2.794490141026062) q[6];
ry(-0.0007557288945466406) q[7];
rz(2.354475111120653) q[7];
ry(-2.8536894627730782) q[8];
rz(0.793883009524059) q[8];
ry(0.0056084556897300075) q[9];
rz(0.3077042855398169) q[9];
ry(-2.6782829551884673) q[10];
rz(-0.03967037132257455) q[10];
ry(1.0499545927699505) q[11];
rz(-1.4423441761222824) q[11];
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
ry(1.3085829103914) q[0];
rz(-2.7593250152046886) q[0];
ry(0.6257822189842095) q[1];
rz(0.6637648754675197) q[1];
ry(1.2597752339546577) q[2];
rz(1.5825278632888375) q[2];
ry(-0.6158518963337185) q[3];
rz(2.289049015802428) q[3];
ry(3.1382848553332616) q[4];
rz(2.3994809517582762) q[4];
ry(-2.525175093150367) q[5];
rz(0.5426355969974127) q[5];
ry(-3.135449362730404) q[6];
rz(-0.3820973294818096) q[6];
ry(-2.492495968352193) q[7];
rz(0.8945333736443795) q[7];
ry(-0.8916107035061702) q[8];
rz(0.1532966867861208) q[8];
ry(2.2482923515525086) q[9];
rz(-0.6495552507287217) q[9];
ry(2.610291695812377) q[10];
rz(-0.20246611480216892) q[10];
ry(-1.1086644539866493) q[11];
rz(1.947898395028142) q[11];
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
ry(-0.3000992354086769) q[0];
rz(-2.4009894117762465) q[0];
ry(-1.6716522067009254) q[1];
rz(0.8681376834674035) q[1];
ry(2.925669930904019) q[2];
rz(1.262756248942828) q[2];
ry(1.1371969746551158) q[3];
rz(1.3964537661905392) q[3];
ry(-0.17523651568416732) q[4];
rz(-2.5667758573479267) q[4];
ry(-3.139828007653591) q[5];
rz(0.11852456616535934) q[5];
ry(-1.5007947858914745) q[6];
rz(-2.262405112782777) q[6];
ry(-3.140807446153803) q[7];
rz(-1.4355476424047555) q[7];
ry(-0.2532123372395811) q[8];
rz(-2.9421041165378417) q[8];
ry(-1.2159639559764448) q[9];
rz(-0.6120000763925839) q[9];
ry(1.415675477702112) q[10];
rz(1.1741874745159657) q[10];
ry(-1.9864994225788593) q[11];
rz(-3.092877164251645) q[11];
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
ry(-1.8796705023813853) q[0];
rz(-0.7911552444892463) q[0];
ry(1.146684031953967) q[1];
rz(-0.9976109329735824) q[1];
ry(-2.5378842057428557) q[2];
rz(2.548710550169929) q[2];
ry(-2.478297460436037) q[3];
rz(-1.5273163645785897) q[3];
ry(3.140623662063128) q[4];
rz(1.021894170661735) q[4];
ry(-1.3735177017340474) q[5];
rz(2.1719691343222682) q[5];
ry(0.00957722451321074) q[6];
rz(0.46621879716641956) q[6];
ry(1.816889638418906) q[7];
rz(0.5350055330925757) q[7];
ry(2.2971143887883634) q[8];
rz(2.71632543790984) q[8];
ry(-2.878289261850644) q[9];
rz(1.5104796119828454) q[9];
ry(0.368195112224331) q[10];
rz(0.5833662124679728) q[10];
ry(2.6120804784448297) q[11];
rz(-2.666314769282392) q[11];
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
ry(1.4745887118631928) q[0];
rz(-0.10191413090136324) q[0];
ry(2.4747917275830162) q[1];
rz(0.8691555407001658) q[1];
ry(-1.4717071257673373) q[2];
rz(-2.104209466475104) q[2];
ry(2.3897894460522657) q[3];
rz(-0.30863703324142033) q[3];
ry(0.09587451429374952) q[4];
rz(-0.5540417383911338) q[4];
ry(-0.002914759443463588) q[5];
rz(-0.7355516891679317) q[5];
ry(-2.5835818614823878) q[6];
rz(-0.43879951447951) q[6];
ry(-3.1415494075137187) q[7];
rz(3.1397320078746604) q[7];
ry(-2.986938671068341) q[8];
rz(-2.9765106413675886) q[8];
ry(2.2900508456070057) q[9];
rz(-0.7259231241165766) q[9];
ry(2.5133017615670847) q[10];
rz(2.5811865121991) q[10];
ry(-1.4305000632903715) q[11];
rz(0.24026062190500405) q[11];
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
ry(3.07120669642767) q[0];
rz(-0.44498792447181224) q[0];
ry(0.6231493687395335) q[1];
rz(-2.998565952626396) q[1];
ry(2.3246769020496494) q[2];
rz(-1.2857613409265265) q[2];
ry(1.9439855962932742) q[3];
rz(-2.1771561229365726) q[3];
ry(0.5865155754358486) q[4];
rz(-0.7657249192616061) q[4];
ry(-2.6680512709009743) q[5];
rz(-1.253452182441312) q[5];
ry(-0.024159290791656574) q[6];
rz(-1.2072493704748748) q[6];
ry(-1.3960974904602022) q[7];
rz(-2.431911668339137) q[7];
ry(-1.8692486411882023) q[8];
rz(-1.1110202354822134) q[8];
ry(0.8774135985362996) q[9];
rz(-1.4127852655018645) q[9];
ry(2.303336244219074) q[10];
rz(-2.072599781851087) q[10];
ry(0.4811083784184805) q[11];
rz(-2.087259106010812) q[11];
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
ry(0.5233408873566567) q[0];
rz(1.5346732875096336) q[0];
ry(-1.054811123842999) q[1];
rz(0.06836816371156029) q[1];
ry(2.871396798524668) q[2];
rz(-1.7743182602853187) q[2];
ry(-1.794100835331352) q[3];
rz(0.933273657610809) q[3];
ry(-2.769678579762474) q[4];
rz(1.8128172743960336) q[4];
ry(3.140786124508117) q[5];
rz(-1.9025057540334673) q[5];
ry(-3.138413857240743) q[6];
rz(2.811739680759093) q[6];
ry(-3.14074788706686) q[7];
rz(0.5194373366136843) q[7];
ry(2.9533891398353345) q[8];
rz(-2.9566338780162433) q[8];
ry(0.3769486329870304) q[9];
rz(-0.4225146546949684) q[9];
ry(0.20866594968805977) q[10];
rz(-1.9747382107909408) q[10];
ry(2.733395280255952) q[11];
rz(-0.7671017735050115) q[11];
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
ry(2.606575264331517) q[0];
rz(2.8406672456333752) q[0];
ry(2.2193398754045064) q[1];
rz(-3.1196685795199373) q[1];
ry(1.3161250298346143) q[2];
rz(-1.163746352591794) q[2];
ry(-2.279524632467944) q[3];
rz(-2.7222425356641122) q[3];
ry(2.4312393819321176) q[4];
rz(2.917841023878604) q[4];
ry(0.39475477893780847) q[5];
rz(-3.0982246117546834) q[5];
ry(-3.137680124891405) q[6];
rz(-0.19874312774361874) q[6];
ry(1.3296524206251032) q[7];
rz(1.7415175066202044) q[7];
ry(0.0557825293663452) q[8];
rz(-0.3930725106564895) q[8];
ry(0.3485813912907811) q[9];
rz(0.5557091513660977) q[9];
ry(1.069319434952698) q[10];
rz(3.037349846140935) q[10];
ry(-0.40548880886611927) q[11];
rz(2.323464293847132) q[11];
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
ry(-1.4885584008723018) q[0];
rz(-0.06107425794679866) q[0];
ry(-2.3215843582047473) q[1];
rz(-2.328906351064753) q[1];
ry(-0.9354321854471221) q[2];
rz(0.6207482896674269) q[2];
ry(0.08793962765368502) q[3];
rz(0.210278867749806) q[3];
ry(-3.0787638393933836) q[4];
rz(2.7519206390959248) q[4];
ry(3.140144526281864) q[5];
rz(0.5066706262948747) q[5];
ry(-0.002270946903860427) q[6];
rz(2.939673865570314) q[6];
ry(0.002425186118537148) q[7];
rz(0.2567569754492738) q[7];
ry(-1.2493547123061282) q[8];
rz(0.8687399181177762) q[8];
ry(0.3506901390837736) q[9];
rz(-0.025304175348708432) q[9];
ry(1.7188858243118341) q[10];
rz(-2.7257371208961874) q[10];
ry(-2.128216272035227) q[11];
rz(0.3078624132816464) q[11];
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
ry(1.6392944799078624) q[0];
rz(-1.3918400775912856) q[0];
ry(-0.8353759989809751) q[1];
rz(-0.9788548908993302) q[1];
ry(-1.4603449861301234) q[2];
rz(1.3846578066158326) q[2];
ry(-2.8933474909991705) q[3];
rz(1.4558237746671656) q[3];
ry(1.0521887923872673) q[4];
rz(0.5058923631540599) q[4];
ry(0.32831662997987454) q[5];
rz(1.7533827348873894) q[5];
ry(-0.004980854850791516) q[6];
rz(-3.0975479385253784) q[6];
ry(-2.337169640581415) q[7];
rz(-0.13223760714719024) q[7];
ry(-3.0349994375218854) q[8];
rz(-1.493969823881039) q[8];
ry(-2.8521537936911376) q[9];
rz(-0.47379204527564805) q[9];
ry(0.340994522060603) q[10];
rz(-2.320477951753581) q[10];
ry(-1.2619036867086244) q[11];
rz(-1.2970613959196484) q[11];
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
ry(-0.1586409249035592) q[0];
rz(-0.431680999736086) q[0];
ry(1.657924822760116) q[1];
rz(2.5325152733570864) q[1];
ry(1.6186718152042636) q[2];
rz(1.7609695072199285) q[2];
ry(-1.9063511233620791) q[3];
rz(-1.8593191252405252) q[3];
ry(-3.0031824387496777) q[4];
rz(2.225421955713161) q[4];
ry(0.008366009189418242) q[5];
rz(0.34924664516201354) q[5];
ry(-0.002034109621668017) q[6];
rz(-0.5368799474469651) q[6];
ry(0.00377284037875636) q[7];
rz(2.9442430675728786) q[7];
ry(2.1636259917326375) q[8];
rz(2.6036229017433814) q[8];
ry(2.6734291859344057) q[9];
rz(0.8994206429680426) q[9];
ry(-2.6060268509579596) q[10];
rz(1.7795018300586998) q[10];
ry(-2.9661688457845328) q[11];
rz(-2.1690905282626716) q[11];
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
ry(-0.9199430702892886) q[0];
rz(1.6125167816047963) q[0];
ry(1.0322293789504435) q[1];
rz(-0.20391386401236708) q[1];
ry(0.5888625436917581) q[2];
rz(-0.5601974051119422) q[2];
ry(2.972803242497138) q[3];
rz(-1.54881934880357) q[3];
ry(-0.09635596013292828) q[4];
rz(0.3036868406065546) q[4];
ry(1.5671848934327735) q[5];
rz(-0.041030625177298684) q[5];
ry(3.137428794079694) q[6];
rz(-2.0127897234896346) q[6];
ry(-3.0696865083380653) q[7];
rz(0.06601443119729479) q[7];
ry(1.631417343525312) q[8];
rz(1.1958670922560994) q[8];
ry(-0.2667683932593922) q[9];
rz(-2.662207170993988) q[9];
ry(0.00233688099877849) q[10];
rz(-1.745216341785312) q[10];
ry(-2.5950573646948984) q[11];
rz(-1.0391290874032733) q[11];
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
ry(2.9074313365357956) q[0];
rz(-0.7911077277644785) q[0];
ry(-1.8362678170235942) q[1];
rz(-1.6660510883392483) q[1];
ry(2.371233897110642) q[2];
rz(-2.6933141109568925) q[2];
ry(2.0485981388995187) q[3];
rz(-1.8201609397644685) q[3];
ry(0.06503041776487639) q[4];
rz(-2.5005469991163487) q[4];
ry(3.1316009129650126) q[5];
rz(2.7741658084060856) q[5];
ry(-3.1413559236247153) q[6];
rz(1.2565839458481773) q[6];
ry(0.0029954831154674444) q[7];
rz(0.3754565031242658) q[7];
ry(3.0067955202558228) q[8];
rz(2.323615103741578) q[8];
ry(1.8231994587852303) q[9];
rz(1.8720903054004079) q[9];
ry(0.4424426104357657) q[10];
rz(0.326746298124089) q[10];
ry(2.9342901879840326) q[11];
rz(-0.4745241262792419) q[11];
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
ry(-1.967469517798209) q[0];
rz(0.7369376693599918) q[0];
ry(2.1667597612718614) q[1];
rz(-1.9359171023478108) q[1];
ry(0.7093888676668261) q[2];
rz(-1.346508020595242) q[2];
ry(-3.0011462452177224) q[3];
rz(-2.204953505380055) q[3];
ry(1.429350650942642) q[4];
rz(1.4680838276787407) q[4];
ry(2.187646298306853) q[5];
rz(0.21814739018986004) q[5];
ry(3.1395964549446753) q[6];
rz(-2.0015904180477158) q[6];
ry(-0.4373267164164785) q[7];
rz(0.7179086739322442) q[7];
ry(-1.426373064849872) q[8];
rz(2.2879229513837678) q[8];
ry(0.15282878692754423) q[9];
rz(-2.367845072641007) q[9];
ry(1.5804042597355448) q[10];
rz(2.757789982759169) q[10];
ry(0.1472328810011776) q[11];
rz(1.2089769063258429) q[11];
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
ry(-0.44388550727311404) q[0];
rz(2.9290644426349592) q[0];
ry(1.326753451526446) q[1];
rz(-1.6554541474142002) q[1];
ry(0.7595388160480172) q[2];
rz(0.47528566187578386) q[2];
ry(0.8831394029167283) q[3];
rz(-0.7724359835381382) q[3];
ry(-1.0610640783533567) q[4];
rz(1.0378375560715205) q[4];
ry(-3.1415120480298246) q[5];
rz(0.8041110137853266) q[5];
ry(0.014807014674396157) q[6];
rz(0.3116132982079884) q[6];
ry(3.1411315263167685) q[7];
rz(-1.2965359505645146) q[7];
ry(2.4691050436137476) q[8];
rz(0.731288347778453) q[8];
ry(1.4685326514473234) q[9];
rz(1.4665215019762847) q[9];
ry(-0.2816695919488055) q[10];
rz(-0.1138129687762514) q[10];
ry(0.99876042530454) q[11];
rz(1.5793277631488634) q[11];
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
ry(1.6173328798514135) q[0];
rz(1.1981089289876967) q[0];
ry(0.774304394270934) q[1];
rz(1.3397202959875276) q[1];
ry(-0.0796402399291578) q[2];
rz(2.2478415766345465) q[2];
ry(-2.036582615599804) q[3];
rz(1.5234771248860213) q[3];
ry(3.0725721975441163) q[4];
rz(0.8859379988734993) q[4];
ry(0.030817719590021882) q[5];
rz(2.1633422326078655) q[5];
ry(0.0022057794753961445) q[6];
rz(1.5554690062904193) q[6];
ry(-2.2550342616332237) q[7];
rz(-0.08265940356333878) q[7];
ry(2.1938889494955003) q[8];
rz(2.7289912672258296) q[8];
ry(1.0419411337217683) q[9];
rz(-2.4969074325621117) q[9];
ry(1.8269589893442764) q[10];
rz(2.254942173165749) q[10];
ry(-1.8157445260683556) q[11];
rz(0.7803983738556992) q[11];
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
ry(2.4934917817648588) q[0];
rz(-0.8589099902396757) q[0];
ry(-1.4888550072932256) q[1];
rz(0.7425072326442548) q[1];
ry(-2.9836323279792967) q[2];
rz(-1.432152557931708) q[2];
ry(-0.7760439620001891) q[3];
rz(2.9738432378633313) q[3];
ry(-2.015916644226917) q[4];
rz(-3.0915479976050917) q[4];
ry(-0.00020917363997163818) q[5];
rz(-1.1265626130814592) q[5];
ry(-0.07204632477618717) q[6];
rz(0.4898151228520537) q[6];
ry(-3.140974982078962) q[7];
rz(1.1892775970159493) q[7];
ry(0.7738878251620438) q[8];
rz(1.4766292966899643) q[8];
ry(1.954477193613773) q[9];
rz(-0.7654011712989218) q[9];
ry(1.2135244493003898) q[10];
rz(1.638957872221197) q[10];
ry(-0.20744836655707127) q[11];
rz(-0.7409000221533807) q[11];
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
ry(-2.535293298829491) q[0];
rz(-1.6375064449126127) q[0];
ry(2.3357358610215107) q[1];
rz(0.5968486001916186) q[1];
ry(-1.4906337819554292) q[2];
rz(-0.9229051185475572) q[2];
ry(-1.029777237866881) q[3];
rz(1.884790125120845) q[3];
ry(0.05068455691539076) q[4];
rz(1.4929438046383297) q[4];
ry(-1.6199945149763295) q[5];
rz(-1.706314854949599) q[5];
ry(0.00024135288263682497) q[6];
rz(2.4986824197350126) q[6];
ry(1.7420079065607392) q[7];
rz(-0.22384666795391261) q[7];
ry(-0.2787592705782007) q[8];
rz(-0.5405227328560696) q[8];
ry(-0.9148296368539999) q[9];
rz(2.3150498060496973) q[9];
ry(-1.8703036814300518) q[10];
rz(1.9962928476855615) q[10];
ry(-2.638313296580368) q[11];
rz(-1.6904561965658047) q[11];
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
ry(-1.2603912555811283) q[0];
rz(1.050452402876859) q[0];
ry(1.9206238021116835) q[1];
rz(1.919099161081796) q[1];
ry(2.636903881365748) q[2];
rz(2.2366181877843734) q[2];
ry(-1.7603370034314239) q[3];
rz(2.8275984674263435) q[3];
ry(-0.07900819638350676) q[4];
rz(1.8424253977881777) q[4];
ry(3.141391757819066) q[5];
rz(-2.020104675951151) q[5];
ry(-3.1404576232300525) q[6];
rz(3.08428572291789) q[6];
ry(-0.0008549744867605189) q[7];
rz(1.8385062429531525) q[7];
ry(-0.32029652090906396) q[8];
rz(-1.3639646641038854) q[8];
ry(1.6226875133861896) q[9];
rz(1.835445121188685) q[9];
ry(-1.6296606457114609) q[10];
rz(-0.9182969622274867) q[10];
ry(2.299365265637762) q[11];
rz(2.3791074080654764) q[11];
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
ry(-2.732088943264742) q[0];
rz(1.688886733581917) q[0];
ry(2.1711398861657463) q[1];
rz(-0.018698900024055387) q[1];
ry(1.1822257931159976) q[2];
rz(2.7372376032490093) q[2];
ry(2.0955914122242048) q[3];
rz(-0.8473485089647205) q[3];
ry(3.0937046167401494) q[4];
rz(2.787704419930161) q[4];
ry(0.9154079854773194) q[5];
rz(-0.7871609398102581) q[5];
ry(0.0006370399807446242) q[6];
rz(0.11371644134758883) q[6];
ry(1.5227312405012832) q[7];
rz(-1.39209754708728) q[7];
ry(2.7100099821198707) q[8];
rz(-0.3200025468669374) q[8];
ry(0.18851005075539018) q[9];
rz(-0.5539164674137993) q[9];
ry(2.9526394096437985) q[10];
rz(-2.2733398830444473) q[10];
ry(1.6580817865969975) q[11];
rz(-1.3610943470953332) q[11];
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
ry(2.51526556406712) q[0];
rz(-1.248237278849477) q[0];
ry(1.508479922790218) q[1];
rz(-0.1757680621665096) q[1];
ry(1.5413962447695602) q[2];
rz(3.110779790673277) q[2];
ry(1.4632332867831925) q[3];
rz(2.1220086964715095) q[3];
ry(0.12950799181637096) q[4];
rz(-2.3380394522722727) q[4];
ry(-0.00014690265441306757) q[5];
rz(-1.5977236646986688) q[5];
ry(-0.2805367216257614) q[6];
rz(2.9663983081163083) q[6];
ry(0.0042231904089460315) q[7];
rz(-1.3501361791959265) q[7];
ry(1.6834255220262828) q[8];
rz(0.5465934482016821) q[8];
ry(-2.634974524647248) q[9];
rz(-1.8024744326907822) q[9];
ry(1.6701780955176406) q[10];
rz(-1.6887337905319306) q[10];
ry(1.9173043725169268) q[11];
rz(2.817989564552511) q[11];
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
ry(0.7760925484852558) q[0];
rz(-0.5090954317315068) q[0];
ry(-0.4178333391261322) q[1];
rz(-2.66886790967741) q[1];
ry(2.608189124726344) q[2];
rz(1.647760681962331) q[2];
ry(1.2596001043493341) q[3];
rz(-1.3295920628641689) q[3];
ry(3.135630367639406) q[4];
rz(-2.6570636285809393) q[4];
ry(-0.7942624909616125) q[5];
rz(1.3613051527134288) q[5];
ry(0.0006595097517498871) q[6];
rz(-1.3473186189459492) q[6];
ry(-3.0976308561439807) q[7];
rz(1.3990666798285334) q[7];
ry(1.5495738489020434) q[8];
rz(2.6975228416003416) q[8];
ry(1.6311946539137845) q[9];
rz(0.3892306657650613) q[9];
ry(-0.6060711852654075) q[10];
rz(-1.4618996432279596) q[10];
ry(2.991436458830646) q[11];
rz(2.126294027286529) q[11];
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
ry(1.8228593071403862) q[0];
rz(-0.11959466342503577) q[0];
ry(-0.8011832069698475) q[1];
rz(0.8605310932980812) q[1];
ry(2.1113675457656216) q[2];
rz(0.9338389958535538) q[2];
ry(-1.6050468369820439) q[3];
rz(-0.8164598124546991) q[3];
ry(3.083513993899124) q[4];
rz(-0.8784800530320505) q[4];
ry(-3.1415364242630406) q[5];
rz(2.029577655723023) q[5];
ry(2.8845729594272576) q[6];
rz(-2.7555913542923585) q[6];
ry(-1.5726671806803851) q[7];
rz(-3.141451014853526) q[7];
ry(-2.60987233919485) q[8];
rz(1.774336415729067) q[8];
ry(1.6254760181644694) q[9];
rz(-3.131314582351047) q[9];
ry(-2.486754557598173) q[10];
rz(1.5862200013380638) q[10];
ry(-1.6781872358122563) q[11];
rz(1.0156124258224746) q[11];
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
ry(2.542862925514638) q[0];
rz(-1.217784113365795) q[0];
ry(-1.0653014770658684) q[1];
rz(0.6883335963830488) q[1];
ry(1.9578411293276992) q[2];
rz(0.09928996653391874) q[2];
ry(0.5742622590849553) q[3];
rz(0.9505467392451737) q[3];
ry(0.004708846458830074) q[4];
rz(-1.189693882303946) q[4];
ry(3.141393315484934) q[5];
rz(-0.33987612256398064) q[5];
ry(-3.140629772611096) q[6];
rz(-2.019363562628783) q[6];
ry(1.5613576096961896) q[7];
rz(-0.0005143331496135772) q[7];
ry(1.5395376407892574) q[8];
rz(0.45102859576697885) q[8];
ry(-1.5639426027421752) q[9];
rz(0.10047848133349112) q[9];
ry(-1.5684669064140015) q[10];
rz(-2.8611142533984184) q[10];
ry(-1.5665783953133994) q[11];
rz(0.018541619293775288) q[11];
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
ry(-1.0298707117806456) q[0];
rz(-0.8170093944650185) q[0];
ry(0.3294566108166546) q[1];
rz(2.2015436748941237) q[1];
ry(2.3854360202168166) q[2];
rz(-1.1677979819048527) q[2];
ry(-1.1761016339378965) q[3];
rz(-0.22935761963591883) q[3];
ry(0.06799984946452486) q[4];
rz(3.0456834156093335) q[4];
ry(-1.5736290245822944) q[5];
rz(-1.5067995562625915) q[5];
ry(0.7863105106037691) q[6];
rz(0.765774289374407) q[6];
ry(-1.703532911898241) q[7];
rz(-1.6956677544805203) q[7];
ry(3.1413865133577135) q[8];
rz(2.220457089427412) q[8];
ry(1.542319283262414) q[9];
rz(-1.5741035589124754) q[9];
ry(2.6288459899083465) q[10];
rz(-1.060213815313001) q[10];
ry(-1.5711761272653826) q[11];
rz(3.120482790961984) q[11];
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
ry(1.9317455016533283) q[0];
rz(-2.635326920514618) q[0];
ry(2.3802668642800584) q[1];
rz(-2.2335875336043403) q[1];
ry(-1.1407500641751422) q[2];
rz(-1.5109472720353043) q[2];
ry(-1.567353597257198) q[3];
rz(-0.11443818181360807) q[3];
ry(-3.090017435106478) q[4];
rz(0.30645374221088145) q[4];
ry(3.1349267498634843) q[5];
rz(1.6353256870950639) q[5];
ry(-3.141038941384539) q[6];
rz(0.6840100455556576) q[6];
ry(-3.138972083650046) q[7];
rz(0.4709348905985786) q[7];
ry(0.005107859368022716) q[8];
rz(-2.18665596516217) q[8];
ry(-1.569228586853791) q[9];
rz(2.6161522538426207) q[9];
ry(0.03649378586475294) q[10];
rz(1.4151316911903975) q[10];
ry(-0.6183253750914787) q[11];
rz(3.1067778550959932) q[11];
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
ry(2.142013393047265) q[0];
rz(2.7815034476110863) q[0];
ry(1.570615733503688) q[1];
rz(3.1093301119249372) q[1];
ry(-0.0003697136531863876) q[2];
rz(-0.8442550897922434) q[2];
ry(2.6293129469327474) q[3];
rz(1.470207173171161) q[3];
ry(0.4216572493311998) q[4];
rz(1.2683971247040402) q[4];
ry(-1.4814736037097926) q[5];
rz(2.9824320620527422) q[5];
ry(1.9708264580075605) q[6];
rz(1.6399630785749748) q[6];
ry(-2.9826550211575436) q[7];
rz(2.4288943114262262) q[7];
ry(2.77646251899661) q[8];
rz(-2.961689492601907) q[8];
ry(-3.0336989235650536) q[9];
rz(-2.9516664971236266) q[9];
ry(2.073983119650337) q[10];
rz(1.477011388443296) q[10];
ry(2.389561550401047) q[11];
rz(-0.06540986235095403) q[11];
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
ry(0.08419460187074268) q[0];
rz(2.53925526425371) q[0];
ry(1.6659554380583756) q[1];
rz(-0.9387468260133617) q[1];
ry(-2.9680694755577335) q[2];
rz(0.8433035967430111) q[2];
ry(-1.5739793127565471) q[3];
rz(-2.4675186847427693) q[3];
ry(-3.138622218426969) q[4];
rz(-0.22846805773242718) q[4];
ry(3.140145694149854) q[5];
rz(0.08518440902457895) q[5];
ry(-0.0002714957586009703) q[6];
rz(-1.6427084686116598) q[6];
ry(0.0007502828384771121) q[7];
rz(0.8348683245803883) q[7];
ry(3.133988198053144) q[8];
rz(1.9713461009813043) q[8];
ry(-3.1412931539643014) q[9];
rz(-1.8758343893261804) q[9];
ry(-0.012257007521823127) q[10];
rz(-2.1799646806845643) q[10];
ry(1.3036430644944383) q[11];
rz(1.0727403893931449) q[11];
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
ry(0.5405922439145349) q[0];
rz(-0.07434889218201413) q[0];
ry(-1.7679206423928076) q[1];
rz(2.4460971067766635) q[1];
ry(-3.023105134143541) q[2];
rz(0.6177096561360749) q[2];
ry(2.4690126637439653) q[3];
rz(1.136871734581336) q[3];
ry(0.8901721165765106) q[4];
rz(0.32779855125687635) q[4];
ry(-2.2071249387574223) q[5];
rz(1.522615416416387) q[5];
ry(2.4298361664714934) q[6];
rz(-1.2177772811550853) q[6];
ry(-1.90506333656311) q[7];
rz(0.29336108620638696) q[7];
ry(2.3087514224574384) q[8];
rz(-2.6935600485839903) q[8];
ry(-1.95913638799098) q[9];
rz(2.0915609830907744) q[9];
ry(0.7148252106945012) q[10];
rz(-1.990584074931963) q[10];
ry(0.47642659505036716) q[11];
rz(-0.45663192615584597) q[11];