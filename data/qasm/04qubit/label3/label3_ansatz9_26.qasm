OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.7597531580739927) q[0];
ry(2.3673157544285797) q[1];
cx q[0],q[1];
ry(1.5138000039392205) q[0];
ry(-0.9616528440095284) q[1];
cx q[0],q[1];
ry(3.118738552817482) q[2];
ry(2.727245741506441) q[3];
cx q[2],q[3];
ry(-1.9616121573639795) q[2];
ry(-0.08177697121027784) q[3];
cx q[2],q[3];
ry(1.5628543033182645) q[0];
ry(-2.9188688285549134) q[2];
cx q[0],q[2];
ry(-2.1058625203960215) q[0];
ry(-1.307770488032296) q[2];
cx q[0],q[2];
ry(0.591672446238622) q[1];
ry(-1.2570331667693766) q[3];
cx q[1],q[3];
ry(1.5967408950327986) q[1];
ry(-1.8853753436383798) q[3];
cx q[1],q[3];
ry(2.396784169168003) q[0];
ry(1.395960571515681) q[3];
cx q[0],q[3];
ry(2.9069488520655127) q[0];
ry(2.559358153698763) q[3];
cx q[0],q[3];
ry(0.16444882542605743) q[1];
ry(2.6179260728459743) q[2];
cx q[1],q[2];
ry(0.012577630662942088) q[1];
ry(0.4633976420206558) q[2];
cx q[1],q[2];
ry(-2.6016592542465466) q[0];
ry(-1.3260959716171081) q[1];
cx q[0],q[1];
ry(-2.058591507480886) q[0];
ry(2.5256204103465163) q[1];
cx q[0],q[1];
ry(-1.9857565900296255) q[2];
ry(2.7432341534491527) q[3];
cx q[2],q[3];
ry(0.05791738186551676) q[2];
ry(0.7999105817227861) q[3];
cx q[2],q[3];
ry(0.9285678083987774) q[0];
ry(-2.4437133307636327) q[2];
cx q[0],q[2];
ry(-0.8355454353940717) q[0];
ry(1.57505560662523) q[2];
cx q[0],q[2];
ry(0.8922442659463184) q[1];
ry(-0.7269626393743884) q[3];
cx q[1],q[3];
ry(-3.1083118908453335) q[1];
ry(-0.27421335121833135) q[3];
cx q[1],q[3];
ry(-1.4464294877705317) q[0];
ry(-1.2380628554238107) q[3];
cx q[0],q[3];
ry(-0.186804029006781) q[0];
ry(1.7169193712483315) q[3];
cx q[0],q[3];
ry(2.6467834904736853) q[1];
ry(-2.7294796914490296) q[2];
cx q[1],q[2];
ry(-0.33154286528211063) q[1];
ry(0.8725135531583259) q[2];
cx q[1],q[2];
ry(1.8711919175703207) q[0];
ry(1.9854492993213506) q[1];
cx q[0],q[1];
ry(1.0955773550689543) q[0];
ry(2.4314069509543113) q[1];
cx q[0],q[1];
ry(2.8012739366354595) q[2];
ry(0.8076809181538174) q[3];
cx q[2],q[3];
ry(-2.368911002491082) q[2];
ry(-0.8067802015203797) q[3];
cx q[2],q[3];
ry(2.1482911390226884) q[0];
ry(-1.314052265461815) q[2];
cx q[0],q[2];
ry(2.6390278146862447) q[0];
ry(-0.9257766125255941) q[2];
cx q[0],q[2];
ry(1.7451670796249381) q[1];
ry(-1.9954108755455993) q[3];
cx q[1],q[3];
ry(2.402173373729212) q[1];
ry(-2.1770400238428134) q[3];
cx q[1],q[3];
ry(2.4542990183977165) q[0];
ry(-2.5920810277075708) q[3];
cx q[0],q[3];
ry(-0.8637662007361069) q[0];
ry(-0.09114337797068453) q[3];
cx q[0],q[3];
ry(-3.0968302771447456) q[1];
ry(0.9836419198307595) q[2];
cx q[1],q[2];
ry(1.525205964252482) q[1];
ry(0.6132388126816385) q[2];
cx q[1],q[2];
ry(-2.6458537671166877) q[0];
ry(-0.7255462512577183) q[1];
cx q[0],q[1];
ry(-3.0508956307830135) q[0];
ry(-1.9588171767347673) q[1];
cx q[0],q[1];
ry(1.8210194174392003) q[2];
ry(2.851871731850484) q[3];
cx q[2],q[3];
ry(-1.3677272860475278) q[2];
ry(-1.6238372673437178) q[3];
cx q[2],q[3];
ry(-1.1304016938924384) q[0];
ry(-1.5837260509244844) q[2];
cx q[0],q[2];
ry(1.9999108544618487) q[0];
ry(0.8430924908399593) q[2];
cx q[0],q[2];
ry(-1.8444736006087687) q[1];
ry(0.9888877256588628) q[3];
cx q[1],q[3];
ry(2.9326154868309224) q[1];
ry(-1.9156790294285047) q[3];
cx q[1],q[3];
ry(2.7157278781121628) q[0];
ry(-1.9746833709125626) q[3];
cx q[0],q[3];
ry(-1.2812509836612653) q[0];
ry(-2.920596541538826) q[3];
cx q[0],q[3];
ry(1.8914949576929647) q[1];
ry(0.4226242993446947) q[2];
cx q[1],q[2];
ry(0.8815941611323387) q[1];
ry(0.8655227144081793) q[2];
cx q[1],q[2];
ry(-0.35800471300981224) q[0];
ry(-0.15177067258792398) q[1];
cx q[0],q[1];
ry(-0.2466654403901514) q[0];
ry(0.37780395043820114) q[1];
cx q[0],q[1];
ry(-0.801091018432876) q[2];
ry(-1.7588500554083204) q[3];
cx q[2],q[3];
ry(1.9837390476891255) q[2];
ry(-1.7316643159411287) q[3];
cx q[2],q[3];
ry(0.6110194265976254) q[0];
ry(0.49838872005784346) q[2];
cx q[0],q[2];
ry(3.0427964391755715) q[0];
ry(1.2701638574726317) q[2];
cx q[0],q[2];
ry(-2.176848599003052) q[1];
ry(-0.42633194685707365) q[3];
cx q[1],q[3];
ry(0.7677257973048158) q[1];
ry(-2.2833695875145277) q[3];
cx q[1],q[3];
ry(0.4089553414306488) q[0];
ry(-1.2771623421849774) q[3];
cx q[0],q[3];
ry(2.1159570803883843) q[0];
ry(1.9999117982114356) q[3];
cx q[0],q[3];
ry(0.04752883060753277) q[1];
ry(0.5530210828462696) q[2];
cx q[1],q[2];
ry(2.1412379985390113) q[1];
ry(-0.01940672305409459) q[2];
cx q[1],q[2];
ry(0.7809097042631166) q[0];
ry(0.15047418587883596) q[1];
cx q[0],q[1];
ry(-2.9080354564748845) q[0];
ry(-2.6242038977267663) q[1];
cx q[0],q[1];
ry(0.9532297294718681) q[2];
ry(-1.909068608795628) q[3];
cx q[2],q[3];
ry(-0.9518691476988881) q[2];
ry(-3.0483625332038176) q[3];
cx q[2],q[3];
ry(0.7073806016085893) q[0];
ry(-1.12781081661529) q[2];
cx q[0],q[2];
ry(-2.7963611327108064) q[0];
ry(1.1784413974063674) q[2];
cx q[0],q[2];
ry(0.2642930956231311) q[1];
ry(-0.05637215065000678) q[3];
cx q[1],q[3];
ry(-1.7819713425408075) q[1];
ry(0.5796953753954668) q[3];
cx q[1],q[3];
ry(0.558373197021177) q[0];
ry(0.03880551598051596) q[3];
cx q[0],q[3];
ry(2.725839843006768) q[0];
ry(-0.8389809516861406) q[3];
cx q[0],q[3];
ry(0.6944979594053802) q[1];
ry(-0.7607824917400192) q[2];
cx q[1],q[2];
ry(-1.3363896212576325) q[1];
ry(1.581996372000515) q[2];
cx q[1],q[2];
ry(2.830336336889182) q[0];
ry(-2.2165165837580743) q[1];
cx q[0],q[1];
ry(2.3420591087352323) q[0];
ry(1.0987475583326782) q[1];
cx q[0],q[1];
ry(-1.0256636443327445) q[2];
ry(-2.7265367217790404) q[3];
cx q[2],q[3];
ry(2.6632741266784534) q[2];
ry(-1.7620815558319043) q[3];
cx q[2],q[3];
ry(2.4217014440504805) q[0];
ry(0.5782711568611849) q[2];
cx q[0],q[2];
ry(1.8254233041998713) q[0];
ry(-0.12072547371528448) q[2];
cx q[0],q[2];
ry(3.129145103970838) q[1];
ry(0.7508664347655999) q[3];
cx q[1],q[3];
ry(-0.9016634781439826) q[1];
ry(3.0509750640198714) q[3];
cx q[1],q[3];
ry(2.053471043944763) q[0];
ry(2.6079039657706757) q[3];
cx q[0],q[3];
ry(0.8141927484898428) q[0];
ry(0.7696526323335864) q[3];
cx q[0],q[3];
ry(2.639076458121909) q[1];
ry(3.0620576917173516) q[2];
cx q[1],q[2];
ry(-2.7270136480415537) q[1];
ry(-2.6547625412322042) q[2];
cx q[1],q[2];
ry(-0.16109479012514516) q[0];
ry(-2.7315579704381956) q[1];
cx q[0],q[1];
ry(-2.226868722732373) q[0];
ry(1.4175264434324415) q[1];
cx q[0],q[1];
ry(-2.775942540021587) q[2];
ry(1.7080345850497796) q[3];
cx q[2],q[3];
ry(0.3412916304741475) q[2];
ry(-1.9711400399495949) q[3];
cx q[2],q[3];
ry(-2.0593066508241535) q[0];
ry(-0.06585168115520457) q[2];
cx q[0],q[2];
ry(-2.116964975916435) q[0];
ry(0.9773007557574482) q[2];
cx q[0],q[2];
ry(1.6566204252007104) q[1];
ry(-1.4487765077744301) q[3];
cx q[1],q[3];
ry(-1.1782141529962873) q[1];
ry(-1.4746585996903283) q[3];
cx q[1],q[3];
ry(1.9554203634843008) q[0];
ry(2.082805992167624) q[3];
cx q[0],q[3];
ry(2.7240927229630407) q[0];
ry(-1.7199156211148001) q[3];
cx q[0],q[3];
ry(0.06757029230067396) q[1];
ry(-3.0523601238694953) q[2];
cx q[1],q[2];
ry(-1.4447293485660504) q[1];
ry(-1.991504680103663) q[2];
cx q[1],q[2];
ry(-0.7403795740150139) q[0];
ry(3.1207391004435667) q[1];
cx q[0],q[1];
ry(-1.2476996084412963) q[0];
ry(-1.414134783699259) q[1];
cx q[0],q[1];
ry(-1.7725708965950417) q[2];
ry(0.6303767127777613) q[3];
cx q[2],q[3];
ry(-0.1625311394011213) q[2];
ry(-1.9619123279254447) q[3];
cx q[2],q[3];
ry(0.9368446246266346) q[0];
ry(-1.4794034439813446) q[2];
cx q[0],q[2];
ry(-1.0239466564846664) q[0];
ry(1.41368692366763) q[2];
cx q[0],q[2];
ry(-2.118035828920699) q[1];
ry(1.0744300095684143) q[3];
cx q[1],q[3];
ry(1.1084281352569538) q[1];
ry(-0.9960514462910841) q[3];
cx q[1],q[3];
ry(-0.46305037996639076) q[0];
ry(-1.2007934006110803) q[3];
cx q[0],q[3];
ry(2.066436757559053) q[0];
ry(0.3581575476116945) q[3];
cx q[0],q[3];
ry(-1.8450380862999758) q[1];
ry(1.2591336558108752) q[2];
cx q[1],q[2];
ry(2.4447250199798907) q[1];
ry(2.042682827318771) q[2];
cx q[1],q[2];
ry(1.1678112910715754) q[0];
ry(-2.6495649573400977) q[1];
cx q[0],q[1];
ry(-0.5093733181570537) q[0];
ry(-1.4520546924393138) q[1];
cx q[0],q[1];
ry(2.2562133993286304) q[2];
ry(-2.2237453228228015) q[3];
cx q[2],q[3];
ry(-0.9537528850766391) q[2];
ry(1.1320586348343564) q[3];
cx q[2],q[3];
ry(1.4765935002506225) q[0];
ry(-1.4452994081981947) q[2];
cx q[0],q[2];
ry(1.7709854927290785) q[0];
ry(-0.017742541804370553) q[2];
cx q[0],q[2];
ry(-0.5199084310451165) q[1];
ry(2.8078087222396584) q[3];
cx q[1],q[3];
ry(-1.4651486414376893) q[1];
ry(-2.681032971490749) q[3];
cx q[1],q[3];
ry(-0.9546982070770332) q[0];
ry(-1.8690370477838956) q[3];
cx q[0],q[3];
ry(0.11385934133323615) q[0];
ry(0.6614251729953922) q[3];
cx q[0],q[3];
ry(-0.2370587156710315) q[1];
ry(-1.6322432009775623) q[2];
cx q[1],q[2];
ry(-1.9040974367824852) q[1];
ry(1.710750804639868) q[2];
cx q[1],q[2];
ry(1.962701391132946) q[0];
ry(-2.4653997907403644) q[1];
cx q[0],q[1];
ry(-0.4365481983627968) q[0];
ry(2.186354820175297) q[1];
cx q[0],q[1];
ry(1.5910166278946853) q[2];
ry(0.7312848257378141) q[3];
cx q[2],q[3];
ry(-0.8366185841502369) q[2];
ry(-1.9785306536220402) q[3];
cx q[2],q[3];
ry(-2.8453961668911574) q[0];
ry(-0.04225345932792113) q[2];
cx q[0],q[2];
ry(0.39481787945095465) q[0];
ry(-2.1935836319211477) q[2];
cx q[0],q[2];
ry(1.635821179816603) q[1];
ry(-2.0262908340006867) q[3];
cx q[1],q[3];
ry(-1.5462193347492637) q[1];
ry(-1.4639792448122924) q[3];
cx q[1],q[3];
ry(-2.2306246062482957) q[0];
ry(-2.3387099866403256) q[3];
cx q[0],q[3];
ry(-1.8675730638532286) q[0];
ry(-1.674847426289898) q[3];
cx q[0],q[3];
ry(-1.6543468640554158) q[1];
ry(1.427751676680801) q[2];
cx q[1],q[2];
ry(2.5616478298797074) q[1];
ry(0.8665368967573155) q[2];
cx q[1],q[2];
ry(-0.41123700739211166) q[0];
ry(2.091169590455875) q[1];
cx q[0],q[1];
ry(0.6032742291745148) q[0];
ry(-1.5267678462498953) q[1];
cx q[0],q[1];
ry(-2.479017533020249) q[2];
ry(2.9722265812615363) q[3];
cx q[2],q[3];
ry(-0.22523305991941722) q[2];
ry(2.5907395915933304) q[3];
cx q[2],q[3];
ry(-0.19721270277369207) q[0];
ry(1.6546043348688197) q[2];
cx q[0],q[2];
ry(-2.272231181190649) q[0];
ry(-2.5204543064076663) q[2];
cx q[0],q[2];
ry(-2.0396608599069705) q[1];
ry(1.6589069809063415) q[3];
cx q[1],q[3];
ry(1.1794958493579202) q[1];
ry(-3.108277426825491) q[3];
cx q[1],q[3];
ry(-2.4364270523878027) q[0];
ry(0.2683764689773813) q[3];
cx q[0],q[3];
ry(-1.559058616648731) q[0];
ry(-2.3231007761704694) q[3];
cx q[0],q[3];
ry(2.923621361860702) q[1];
ry(1.5199668048188224) q[2];
cx q[1],q[2];
ry(1.2471259524549472) q[1];
ry(2.9599370863838654) q[2];
cx q[1],q[2];
ry(2.7990492464087002) q[0];
ry(3.08139374363773) q[1];
cx q[0],q[1];
ry(0.45355680952114535) q[0];
ry(0.7331900133825023) q[1];
cx q[0],q[1];
ry(-2.583568906414834) q[2];
ry(2.2299230911044514) q[3];
cx q[2],q[3];
ry(1.0290632647154219) q[2];
ry(-2.297491707792313) q[3];
cx q[2],q[3];
ry(-1.9650279464984566) q[0];
ry(-0.8869549280813269) q[2];
cx q[0],q[2];
ry(0.8066630488130812) q[0];
ry(-0.5383123674361094) q[2];
cx q[0],q[2];
ry(3.0320673144303316) q[1];
ry(-2.059390873446187) q[3];
cx q[1],q[3];
ry(3.06789146399307) q[1];
ry(-2.7210715810932378) q[3];
cx q[1],q[3];
ry(-2.419702797719662) q[0];
ry(-1.4833466358480651) q[3];
cx q[0],q[3];
ry(-0.9599225039528019) q[0];
ry(1.2682906044431703) q[3];
cx q[0],q[3];
ry(-1.435664872517586) q[1];
ry(1.2276125908471123) q[2];
cx q[1],q[2];
ry(-1.1206706382131462) q[1];
ry(-0.896221117446901) q[2];
cx q[1],q[2];
ry(1.8373747904959128) q[0];
ry(1.4322537366759063) q[1];
cx q[0],q[1];
ry(1.521996290274057) q[0];
ry(2.2058790462425937) q[1];
cx q[0],q[1];
ry(-1.6498369320249888) q[2];
ry(1.414576302521525) q[3];
cx q[2],q[3];
ry(1.9599402258975385) q[2];
ry(0.5737178972125685) q[3];
cx q[2],q[3];
ry(1.6423725359303047) q[0];
ry(-2.3541255875784737) q[2];
cx q[0],q[2];
ry(-0.9733782831702724) q[0];
ry(0.6509112951475035) q[2];
cx q[0],q[2];
ry(-1.306247099526538) q[1];
ry(1.7079651149923607) q[3];
cx q[1],q[3];
ry(2.6941665839342175) q[1];
ry(2.7600310000415096) q[3];
cx q[1],q[3];
ry(1.4747463502332332) q[0];
ry(-0.8757981840480724) q[3];
cx q[0],q[3];
ry(0.6180577280180044) q[0];
ry(2.954351300897566) q[3];
cx q[0],q[3];
ry(-0.24952195776992572) q[1];
ry(-0.7817668123458833) q[2];
cx q[1],q[2];
ry(-2.532726855411753) q[1];
ry(-0.48957335690518977) q[2];
cx q[1],q[2];
ry(2.6837269458668467) q[0];
ry(-2.871261419103059) q[1];
cx q[0],q[1];
ry(-1.8817257361619455) q[0];
ry(-0.8085588187337046) q[1];
cx q[0],q[1];
ry(-0.7476589118759204) q[2];
ry(-0.12238155727697997) q[3];
cx q[2],q[3];
ry(1.0575798916429582) q[2];
ry(-2.0233295734954604) q[3];
cx q[2],q[3];
ry(-2.0018298386135416) q[0];
ry(1.2566747370223519) q[2];
cx q[0],q[2];
ry(2.7702932131592433) q[0];
ry(2.6413854852957255) q[2];
cx q[0],q[2];
ry(3.1001765533271097) q[1];
ry(-0.1312679468747131) q[3];
cx q[1],q[3];
ry(-0.2947336228387064) q[1];
ry(1.6997913481845623) q[3];
cx q[1],q[3];
ry(-2.7673165772413273) q[0];
ry(-1.468469625556195) q[3];
cx q[0],q[3];
ry(0.7136160580861042) q[0];
ry(-1.0320652491852023) q[3];
cx q[0],q[3];
ry(0.2968277628760656) q[1];
ry(1.6965871237756387) q[2];
cx q[1],q[2];
ry(-0.7497888448991752) q[1];
ry(-2.8934111752102636) q[2];
cx q[1],q[2];
ry(-2.636844518028295) q[0];
ry(1.0768029493840965) q[1];
cx q[0],q[1];
ry(-0.6916514608122942) q[0];
ry(0.9399268030807413) q[1];
cx q[0],q[1];
ry(-0.9657935351958455) q[2];
ry(0.3522605591968682) q[3];
cx q[2],q[3];
ry(-1.4526161558779767) q[2];
ry(-0.41658517269684897) q[3];
cx q[2],q[3];
ry(-0.553191406883955) q[0];
ry(1.593025560431002) q[2];
cx q[0],q[2];
ry(-1.1092924463083118) q[0];
ry(-0.5821964389829118) q[2];
cx q[0],q[2];
ry(-0.5033671059132026) q[1];
ry(-1.6717314109875734) q[3];
cx q[1],q[3];
ry(0.6685436315003148) q[1];
ry(-1.481289208532045) q[3];
cx q[1],q[3];
ry(-2.884212567883267) q[0];
ry(2.709780100257134) q[3];
cx q[0],q[3];
ry(1.9491147606573955) q[0];
ry(-0.7380161388028537) q[3];
cx q[0],q[3];
ry(-2.8667299779837436) q[1];
ry(1.8837986226758616) q[2];
cx q[1],q[2];
ry(0.08332265173518037) q[1];
ry(-0.977431693659307) q[2];
cx q[1],q[2];
ry(-0.20790915706812382) q[0];
ry(-0.07639384169275586) q[1];
cx q[0],q[1];
ry(-1.7951337385791453) q[0];
ry(-0.731202471298306) q[1];
cx q[0],q[1];
ry(-2.6902575498214505) q[2];
ry(2.4012351509396916) q[3];
cx q[2],q[3];
ry(-0.9858985694916731) q[2];
ry(1.7356023365208264) q[3];
cx q[2],q[3];
ry(0.2636989493399491) q[0];
ry(-1.3964044754114404) q[2];
cx q[0],q[2];
ry(-2.2853407992235666) q[0];
ry(-2.4387276821372503) q[2];
cx q[0],q[2];
ry(2.0584304542818774) q[1];
ry(-3.1026869228947183) q[3];
cx q[1],q[3];
ry(0.6603912721769252) q[1];
ry(1.285825450260396) q[3];
cx q[1],q[3];
ry(-1.2387930238750469) q[0];
ry(-1.986003338537019) q[3];
cx q[0],q[3];
ry(-2.157818207490048) q[0];
ry(1.874505382758585) q[3];
cx q[0],q[3];
ry(-2.1059464523241958) q[1];
ry(2.28483949746162) q[2];
cx q[1],q[2];
ry(-1.1982759765329438) q[1];
ry(2.231860948525463) q[2];
cx q[1],q[2];
ry(-2.8763370563117365) q[0];
ry(-0.48907707493089525) q[1];
cx q[0],q[1];
ry(-2.7048607040765846) q[0];
ry(2.320321011090885) q[1];
cx q[0],q[1];
ry(1.4203438867711036) q[2];
ry(-0.9075871963289989) q[3];
cx q[2],q[3];
ry(1.378709218605912) q[2];
ry(0.0386066678935837) q[3];
cx q[2],q[3];
ry(2.074980209252407) q[0];
ry(-1.3460736346538567) q[2];
cx q[0],q[2];
ry(1.0228224798765944) q[0];
ry(1.2923370934314145) q[2];
cx q[0],q[2];
ry(0.49709785473045714) q[1];
ry(2.949938426416489) q[3];
cx q[1],q[3];
ry(2.19106447372698) q[1];
ry(-0.6436141413410743) q[3];
cx q[1],q[3];
ry(1.0879525273095265) q[0];
ry(-1.1007250559548707) q[3];
cx q[0],q[3];
ry(1.7501769249808814) q[0];
ry(-0.3631258032096165) q[3];
cx q[0],q[3];
ry(-0.06314213372076782) q[1];
ry(-2.7323307687973792) q[2];
cx q[1],q[2];
ry(-2.9880102716025063) q[1];
ry(2.196339968654555) q[2];
cx q[1],q[2];
ry(-2.736229265489868) q[0];
ry(1.8023406117691882) q[1];
cx q[0],q[1];
ry(-1.5160434954591633) q[0];
ry(-1.4562031566850937) q[1];
cx q[0],q[1];
ry(-2.538694190802903) q[2];
ry(-0.7859920295389512) q[3];
cx q[2],q[3];
ry(-0.9360915156265156) q[2];
ry(-2.3791576804293206) q[3];
cx q[2],q[3];
ry(-2.275086125647386) q[0];
ry(1.1111107349238418) q[2];
cx q[0],q[2];
ry(1.7283879992662283) q[0];
ry(0.923671813635722) q[2];
cx q[0],q[2];
ry(2.394410097063524) q[1];
ry(1.328182574215674) q[3];
cx q[1],q[3];
ry(-2.1859048641658934) q[1];
ry(-1.5835540002465824) q[3];
cx q[1],q[3];
ry(0.9523144907950005) q[0];
ry(1.1222385465334062) q[3];
cx q[0],q[3];
ry(1.6626141231177387) q[0];
ry(2.7646479922573794) q[3];
cx q[0],q[3];
ry(0.8323358868111529) q[1];
ry(2.82819647228263) q[2];
cx q[1],q[2];
ry(-2.455674234102779) q[1];
ry(-2.664184791854701) q[2];
cx q[1],q[2];
ry(0.22847595147749453) q[0];
ry(0.050991983129894614) q[1];
cx q[0],q[1];
ry(1.4843793296614296) q[0];
ry(0.05106493285580971) q[1];
cx q[0],q[1];
ry(-1.205391864245346) q[2];
ry(-1.826736991454516) q[3];
cx q[2],q[3];
ry(-2.7065109715830906) q[2];
ry(-2.930217767120257) q[3];
cx q[2],q[3];
ry(1.2013184444527178) q[0];
ry(2.4787670409707685) q[2];
cx q[0],q[2];
ry(0.45593189486880087) q[0];
ry(2.2678928933339613) q[2];
cx q[0],q[2];
ry(0.11380083861088443) q[1];
ry(2.4086036635973396) q[3];
cx q[1],q[3];
ry(-1.123055628648569) q[1];
ry(-0.8923481072348209) q[3];
cx q[1],q[3];
ry(2.9852404538359862) q[0];
ry(-0.052621857043352975) q[3];
cx q[0],q[3];
ry(1.622875801085004) q[0];
ry(2.664486224905583) q[3];
cx q[0],q[3];
ry(-2.8850940126831426) q[1];
ry(1.5794592482259635) q[2];
cx q[1],q[2];
ry(0.1219869183939597) q[1];
ry(0.7053510592632417) q[2];
cx q[1],q[2];
ry(0.81971181299643) q[0];
ry(1.0922870805400342) q[1];
cx q[0],q[1];
ry(-2.2212492015204766) q[0];
ry(-1.8056221894470017) q[1];
cx q[0],q[1];
ry(0.4585830891322154) q[2];
ry(0.45124111874708284) q[3];
cx q[2],q[3];
ry(2.557113212108294) q[2];
ry(1.2275586743967537) q[3];
cx q[2],q[3];
ry(1.723924431966771) q[0];
ry(-2.6260949549825305) q[2];
cx q[0],q[2];
ry(1.0004365384908542) q[0];
ry(-2.704536506148938) q[2];
cx q[0],q[2];
ry(2.017049967173574) q[1];
ry(0.8882269598134515) q[3];
cx q[1],q[3];
ry(-1.0494853580953205) q[1];
ry(2.9516434453634433) q[3];
cx q[1],q[3];
ry(0.8948578964380764) q[0];
ry(-0.5885079134832081) q[3];
cx q[0],q[3];
ry(-2.9147256993960666) q[0];
ry(1.831906536990358) q[3];
cx q[0],q[3];
ry(-0.09298113777841888) q[1];
ry(2.812174194875238) q[2];
cx q[1],q[2];
ry(-2.4430993966437624) q[1];
ry(1.4751244885340862) q[2];
cx q[1],q[2];
ry(-0.2853384590043726) q[0];
ry(1.528405617988953) q[1];
cx q[0],q[1];
ry(-0.4865313445541611) q[0];
ry(2.7199929072584554) q[1];
cx q[0],q[1];
ry(-1.7014833450433724) q[2];
ry(-1.9175752561871109) q[3];
cx q[2],q[3];
ry(-1.78375441151011) q[2];
ry(-2.774398552806838) q[3];
cx q[2],q[3];
ry(-1.7825769190223664) q[0];
ry(1.281731776498279) q[2];
cx q[0],q[2];
ry(0.12851391502403417) q[0];
ry(-1.7919912873922086) q[2];
cx q[0],q[2];
ry(-2.8403288510644784) q[1];
ry(-2.8033273712127387) q[3];
cx q[1],q[3];
ry(-2.2260430554054924) q[1];
ry(-1.2608079480797938) q[3];
cx q[1],q[3];
ry(-0.5457000982839466) q[0];
ry(-1.311793607708215) q[3];
cx q[0],q[3];
ry(0.6738656188106961) q[0];
ry(-1.1638622438602402) q[3];
cx q[0],q[3];
ry(-1.0335536219636943) q[1];
ry(0.7158110224945923) q[2];
cx q[1],q[2];
ry(1.422226040876767) q[1];
ry(3.0943644128728813) q[2];
cx q[1],q[2];
ry(0.9192433016000012) q[0];
ry(0.6040518145359801) q[1];
cx q[0],q[1];
ry(-0.428329353000273) q[0];
ry(2.859464327310045) q[1];
cx q[0],q[1];
ry(1.9895594953226876) q[2];
ry(-1.5099062225544286) q[3];
cx q[2],q[3];
ry(1.090626108009177) q[2];
ry(-2.1482614971497576) q[3];
cx q[2],q[3];
ry(3.109287668373077) q[0];
ry(-0.3723767628254223) q[2];
cx q[0],q[2];
ry(-0.1880187928319641) q[0];
ry(0.6634113260703742) q[2];
cx q[0],q[2];
ry(-2.161733246760762) q[1];
ry(0.9785205249954422) q[3];
cx q[1],q[3];
ry(0.9766717429857534) q[1];
ry(0.9738950188807509) q[3];
cx q[1],q[3];
ry(-1.1441121289062812) q[0];
ry(2.2769388000612887) q[3];
cx q[0],q[3];
ry(0.137190054425033) q[0];
ry(-1.3898580401805145) q[3];
cx q[0],q[3];
ry(1.311395347843681) q[1];
ry(-0.1452879391921538) q[2];
cx q[1],q[2];
ry(-1.3010377879447514) q[1];
ry(0.31285037071791744) q[2];
cx q[1],q[2];
ry(-2.4926636258772055) q[0];
ry(-1.721882301221937) q[1];
cx q[0],q[1];
ry(1.8781072091885413) q[0];
ry(-1.0976660516194183) q[1];
cx q[0],q[1];
ry(-1.8728746609567883) q[2];
ry(-1.4201364954180953) q[3];
cx q[2],q[3];
ry(-2.0928887187558214) q[2];
ry(3.1240485315215185) q[3];
cx q[2],q[3];
ry(-2.1171033273747693) q[0];
ry(1.7563654817433374) q[2];
cx q[0],q[2];
ry(2.8365254928729793) q[0];
ry(-0.5115067965119745) q[2];
cx q[0],q[2];
ry(1.0236675831314246) q[1];
ry(0.5504772900079811) q[3];
cx q[1],q[3];
ry(1.0461686354365298) q[1];
ry(-0.24659257648785135) q[3];
cx q[1],q[3];
ry(0.9443798425208927) q[0];
ry(-0.10063081595967929) q[3];
cx q[0],q[3];
ry(1.650489205712674) q[0];
ry(2.451893952093975) q[3];
cx q[0],q[3];
ry(2.684049148429021) q[1];
ry(1.7546545852859745) q[2];
cx q[1],q[2];
ry(-0.7108053631766253) q[1];
ry(1.0024142719804638) q[2];
cx q[1],q[2];
ry(-0.08514275682378614) q[0];
ry(-2.8174782114629355) q[1];
cx q[0],q[1];
ry(-2.5175844647112777) q[0];
ry(-2.502777844827198) q[1];
cx q[0],q[1];
ry(-2.0487993523613044) q[2];
ry(-0.08539049976008188) q[3];
cx q[2],q[3];
ry(2.1209356485482176) q[2];
ry(-2.3681004468317672) q[3];
cx q[2],q[3];
ry(2.5412343577280794) q[0];
ry(-2.322357172261961) q[2];
cx q[0],q[2];
ry(-2.5710634013589337) q[0];
ry(2.15922131392706) q[2];
cx q[0],q[2];
ry(-2.804692031092402) q[1];
ry(2.5494350614835866) q[3];
cx q[1],q[3];
ry(2.831656086600483) q[1];
ry(-1.5889832133214281) q[3];
cx q[1],q[3];
ry(3.08618648690703) q[0];
ry(-0.47869358025073416) q[3];
cx q[0],q[3];
ry(-0.34876487733557493) q[0];
ry(-2.412917722322569) q[3];
cx q[0],q[3];
ry(1.3725666050818477) q[1];
ry(1.613109496901019) q[2];
cx q[1],q[2];
ry(0.6513602468179549) q[1];
ry(2.072607473141897) q[2];
cx q[1],q[2];
ry(-0.4910223409149905) q[0];
ry(0.30482233729510966) q[1];
cx q[0],q[1];
ry(0.2836909063812753) q[0];
ry(-0.6696050175172577) q[1];
cx q[0],q[1];
ry(2.2583429666304915) q[2];
ry(1.1848559455578815) q[3];
cx q[2],q[3];
ry(-1.5274844017780487) q[2];
ry(-2.092917197911625) q[3];
cx q[2],q[3];
ry(2.2240340589265104) q[0];
ry(2.1031611096164804) q[2];
cx q[0],q[2];
ry(0.16244588586903055) q[0];
ry(0.5774651974740168) q[2];
cx q[0],q[2];
ry(-2.995675899435339) q[1];
ry(-1.7488880551997938) q[3];
cx q[1],q[3];
ry(-2.1960310966397) q[1];
ry(-2.978717734051935) q[3];
cx q[1],q[3];
ry(-0.10563641230969445) q[0];
ry(-2.639562682595153) q[3];
cx q[0],q[3];
ry(1.9782250379108692) q[0];
ry(1.7993922636112858) q[3];
cx q[0],q[3];
ry(1.2035562738597407) q[1];
ry(2.923990210528185) q[2];
cx q[1],q[2];
ry(-1.991739858778936) q[1];
ry(-0.6225783114788962) q[2];
cx q[1],q[2];
ry(-0.629313771531995) q[0];
ry(-2.3403318942137368) q[1];
cx q[0],q[1];
ry(-0.9677665856190751) q[0];
ry(-2.606622357613536) q[1];
cx q[0],q[1];
ry(-2.0285035490289407) q[2];
ry(1.5735614488294447) q[3];
cx q[2],q[3];
ry(-0.8576926524089616) q[2];
ry(-2.0901551644998078) q[3];
cx q[2],q[3];
ry(-0.5040836582278524) q[0];
ry(1.068416406922827) q[2];
cx q[0],q[2];
ry(-2.804078446803668) q[0];
ry(2.0000047941809536) q[2];
cx q[0],q[2];
ry(-1.6678529010356344) q[1];
ry(1.2491398548629504) q[3];
cx q[1],q[3];
ry(-0.7980402800950257) q[1];
ry(0.42846640811569614) q[3];
cx q[1],q[3];
ry(-2.576871750890899) q[0];
ry(-1.8236001449430876) q[3];
cx q[0],q[3];
ry(-0.21638551922182495) q[0];
ry(-2.964308593204878) q[3];
cx q[0],q[3];
ry(-1.6239629743364423) q[1];
ry(-0.9653849285343261) q[2];
cx q[1],q[2];
ry(-2.8754936967886473) q[1];
ry(-1.5139621986038119) q[2];
cx q[1],q[2];
ry(-1.7828945660137945) q[0];
ry(-0.38256644254205324) q[1];
cx q[0],q[1];
ry(0.3994847959158463) q[0];
ry(1.788016716541582) q[1];
cx q[0],q[1];
ry(0.3361409332830245) q[2];
ry(-0.7168298304006308) q[3];
cx q[2],q[3];
ry(1.5423390381659283) q[2];
ry(3.1157589218349067) q[3];
cx q[2],q[3];
ry(-2.135700445715681) q[0];
ry(-1.5596372660454287) q[2];
cx q[0],q[2];
ry(-2.5508158798725473) q[0];
ry(-2.020347342318976) q[2];
cx q[0],q[2];
ry(1.5837569777138047) q[1];
ry(1.6013538949345953) q[3];
cx q[1],q[3];
ry(-0.9916761620105303) q[1];
ry(1.2004102107467098) q[3];
cx q[1],q[3];
ry(0.9635067460165468) q[0];
ry(0.5287562064380521) q[3];
cx q[0],q[3];
ry(-0.03349755696592993) q[0];
ry(1.5372549928846384) q[3];
cx q[0],q[3];
ry(-3.0307765010772645) q[1];
ry(0.21361999565030199) q[2];
cx q[1],q[2];
ry(2.005444249705747) q[1];
ry(2.0382031940752388) q[2];
cx q[1],q[2];
ry(-2.7053579434935027) q[0];
ry(-0.1035850819927993) q[1];
cx q[0],q[1];
ry(3.0148424475833604) q[0];
ry(0.6909828620496885) q[1];
cx q[0],q[1];
ry(-0.6533036575159006) q[2];
ry(2.1687583408162614) q[3];
cx q[2],q[3];
ry(2.523422466232964) q[2];
ry(2.959203178144149) q[3];
cx q[2],q[3];
ry(-2.4824292291850103) q[0];
ry(3.1263624516585207) q[2];
cx q[0],q[2];
ry(1.5881192567537843) q[0];
ry(3.0673871110091193) q[2];
cx q[0],q[2];
ry(-0.9906726176901097) q[1];
ry(-2.363571823282147) q[3];
cx q[1],q[3];
ry(-2.907080734077196) q[1];
ry(2.092943853707312) q[3];
cx q[1],q[3];
ry(2.7584136454743025) q[0];
ry(1.586591084841876) q[3];
cx q[0],q[3];
ry(0.42898728777940365) q[0];
ry(-0.7956640669567331) q[3];
cx q[0],q[3];
ry(-0.8672481454532743) q[1];
ry(2.076620523139175) q[2];
cx q[1],q[2];
ry(1.8651838871448987) q[1];
ry(2.487410593963561) q[2];
cx q[1],q[2];
ry(0.6883070507129377) q[0];
ry(2.910205842947834) q[1];
ry(-2.8610150173426723) q[2];
ry(0.4500995372670857) q[3];