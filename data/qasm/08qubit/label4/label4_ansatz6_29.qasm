OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.2471308054296602) q[0];
ry(-0.6435489236773133) q[1];
cx q[0],q[1];
ry(-0.46276048858122376) q[0];
ry(-0.4604539091463167) q[1];
cx q[0],q[1];
ry(-2.185590013097208) q[1];
ry(-2.0165610673594916) q[2];
cx q[1],q[2];
ry(1.8058510495215092) q[1];
ry(-1.086193185010044) q[2];
cx q[1],q[2];
ry(2.2660460050979263) q[2];
ry(-0.11679277400380172) q[3];
cx q[2],q[3];
ry(1.4281727447171666) q[2];
ry(1.3562951297758914) q[3];
cx q[2],q[3];
ry(1.9903015289816894) q[3];
ry(-1.9642098107048653) q[4];
cx q[3],q[4];
ry(-1.7387042880909114) q[3];
ry(-1.8676654564611406) q[4];
cx q[3],q[4];
ry(1.9194854827982848) q[4];
ry(1.7231554420623736) q[5];
cx q[4],q[5];
ry(-1.1748122051406167) q[4];
ry(0.7599946834221482) q[5];
cx q[4],q[5];
ry(-1.1541663400517068) q[5];
ry(-0.8301699059578492) q[6];
cx q[5],q[6];
ry(-2.7112432882795474) q[5];
ry(-2.9586948585880006) q[6];
cx q[5],q[6];
ry(2.694792852540425) q[6];
ry(-2.643926843828918) q[7];
cx q[6],q[7];
ry(-1.3574161930387858) q[6];
ry(-0.3261670965576057) q[7];
cx q[6],q[7];
ry(-2.2592109819564277) q[0];
ry(0.4184327561356796) q[1];
cx q[0],q[1];
ry(-2.4699643868554326) q[0];
ry(1.790761757515517) q[1];
cx q[0],q[1];
ry(2.9026865166582785) q[1];
ry(0.41705559917088764) q[2];
cx q[1],q[2];
ry(2.4360353037191795) q[1];
ry(2.704406282098228) q[2];
cx q[1],q[2];
ry(3.0201285475141946) q[2];
ry(-0.08310251242726441) q[3];
cx q[2],q[3];
ry(0.5996805714706159) q[2];
ry(2.4222364639114304) q[3];
cx q[2],q[3];
ry(2.8916222539500245) q[3];
ry(1.248060858727876) q[4];
cx q[3],q[4];
ry(0.24588806411287403) q[3];
ry(0.05130649501724379) q[4];
cx q[3],q[4];
ry(0.31131222585066237) q[4];
ry(-2.7447151587388277) q[5];
cx q[4],q[5];
ry(-0.5264463935816227) q[4];
ry(-2.815357885957505) q[5];
cx q[4],q[5];
ry(-2.6742671859463996) q[5];
ry(3.0818783684537716) q[6];
cx q[5],q[6];
ry(-2.0900138581875103) q[5];
ry(1.7940251368021978) q[6];
cx q[5],q[6];
ry(-1.261336529675532) q[6];
ry(-1.0439676526599433) q[7];
cx q[6],q[7];
ry(2.5327986825557174) q[6];
ry(2.0105608807166147) q[7];
cx q[6],q[7];
ry(-0.37336370811934266) q[0];
ry(3.0208242286317497) q[1];
cx q[0],q[1];
ry(1.3461957208311701) q[0];
ry(-1.960441158819541) q[1];
cx q[0],q[1];
ry(-0.905725373853838) q[1];
ry(-2.583789278240679) q[2];
cx q[1],q[2];
ry(1.11351214385318) q[1];
ry(2.5074098663709132) q[2];
cx q[1],q[2];
ry(-2.402261804467813) q[2];
ry(-0.7719234282706271) q[3];
cx q[2],q[3];
ry(-0.665835477123558) q[2];
ry(1.5288290877595179) q[3];
cx q[2],q[3];
ry(-1.636046096688383) q[3];
ry(0.6172867724534031) q[4];
cx q[3],q[4];
ry(0.4847642572929258) q[3];
ry(-2.8644251666813223) q[4];
cx q[3],q[4];
ry(2.125277348594965) q[4];
ry(2.3631908910823007) q[5];
cx q[4],q[5];
ry(1.7569195453212512) q[4];
ry(0.49970448655964267) q[5];
cx q[4],q[5];
ry(-1.2661218576028122) q[5];
ry(1.2776143537207743) q[6];
cx q[5],q[6];
ry(1.2962044695005568) q[5];
ry(-3.0850513763611325) q[6];
cx q[5],q[6];
ry(1.6321295498981006) q[6];
ry(-2.0572197111191954) q[7];
cx q[6],q[7];
ry(0.34126073611125257) q[6];
ry(-3.123248373574269) q[7];
cx q[6],q[7];
ry(-1.534624142067499) q[0];
ry(-2.2291712300800617) q[1];
cx q[0],q[1];
ry(2.7416323934376954) q[0];
ry(-1.6092517630856469) q[1];
cx q[0],q[1];
ry(-0.9519984205599198) q[1];
ry(-0.9374405167182981) q[2];
cx q[1],q[2];
ry(0.7394362765387513) q[1];
ry(1.2729429689979508) q[2];
cx q[1],q[2];
ry(0.9548624814414287) q[2];
ry(0.14917201700503924) q[3];
cx q[2],q[3];
ry(1.926626344749672) q[2];
ry(-1.0305067026117083) q[3];
cx q[2],q[3];
ry(0.7879939449413085) q[3];
ry(-1.4690099901724771) q[4];
cx q[3],q[4];
ry(1.3890850567404502) q[3];
ry(2.5639473911980137) q[4];
cx q[3],q[4];
ry(2.556479858828862) q[4];
ry(2.2685246982782696) q[5];
cx q[4],q[5];
ry(-1.4938503916789276) q[4];
ry(-1.4707303577497441) q[5];
cx q[4],q[5];
ry(-2.5434328712754857) q[5];
ry(-2.696451513260203) q[6];
cx q[5],q[6];
ry(-1.2916388363168876) q[5];
ry(-2.0357898297658594) q[6];
cx q[5],q[6];
ry(2.7080559659597143) q[6];
ry(2.4827177325177425) q[7];
cx q[6],q[7];
ry(-2.4205136889818735) q[6];
ry(2.7246772588742205) q[7];
cx q[6],q[7];
ry(1.463852944885918) q[0];
ry(1.4669142713179082) q[1];
cx q[0],q[1];
ry(-1.5671095769743673) q[0];
ry(-0.17264302942620446) q[1];
cx q[0],q[1];
ry(-2.0634927013235114) q[1];
ry(1.065411046262545) q[2];
cx q[1],q[2];
ry(1.6978297067752584) q[1];
ry(1.100067963610095) q[2];
cx q[1],q[2];
ry(-2.374917627439889) q[2];
ry(-0.05740692660443902) q[3];
cx q[2],q[3];
ry(0.9362885124855902) q[2];
ry(-2.7006423714836583) q[3];
cx q[2],q[3];
ry(-0.5662820580691298) q[3];
ry(2.8684758710457854) q[4];
cx q[3],q[4];
ry(0.6856072105719223) q[3];
ry(-0.710369658002378) q[4];
cx q[3],q[4];
ry(0.30058939444657273) q[4];
ry(-0.1917284675293425) q[5];
cx q[4],q[5];
ry(-1.7547294342485058) q[4];
ry(1.9909424748652071) q[5];
cx q[4],q[5];
ry(-0.24321883984490977) q[5];
ry(-1.7654146820578482) q[6];
cx q[5],q[6];
ry(1.1831791113458268) q[5];
ry(2.099439195288994) q[6];
cx q[5],q[6];
ry(-0.40399118374488013) q[6];
ry(2.189660804563831) q[7];
cx q[6],q[7];
ry(0.8419518440399174) q[6];
ry(-2.551452174551507) q[7];
cx q[6],q[7];
ry(-0.33078711113787307) q[0];
ry(-0.5516401811444158) q[1];
cx q[0],q[1];
ry(1.6877135558835394) q[0];
ry(3.0451841938349324) q[1];
cx q[0],q[1];
ry(1.7022710800283207) q[1];
ry(2.354027785634783) q[2];
cx q[1],q[2];
ry(2.4896131276293607) q[1];
ry(2.104628625318025) q[2];
cx q[1],q[2];
ry(-0.7308250491092996) q[2];
ry(-0.6869735667398564) q[3];
cx q[2],q[3];
ry(1.9626770406843468) q[2];
ry(1.0532706364953344) q[3];
cx q[2],q[3];
ry(-0.8696833720624033) q[3];
ry(-1.2484122540455092) q[4];
cx q[3],q[4];
ry(2.3248795427533286) q[3];
ry(-1.574284651802851) q[4];
cx q[3],q[4];
ry(0.2147182361018392) q[4];
ry(-3.104818293317763) q[5];
cx q[4],q[5];
ry(-2.5094144994547647) q[4];
ry(0.892461876690434) q[5];
cx q[4],q[5];
ry(-2.040774784311883) q[5];
ry(-2.604097591401546) q[6];
cx q[5],q[6];
ry(0.7094096916153942) q[5];
ry(1.2597751657452514) q[6];
cx q[5],q[6];
ry(2.331157284208305) q[6];
ry(-0.3178680075521827) q[7];
cx q[6],q[7];
ry(-2.0357621107784825) q[6];
ry(1.0984171717280145) q[7];
cx q[6],q[7];
ry(3.13549861871181) q[0];
ry(-1.7449646028520984) q[1];
cx q[0],q[1];
ry(1.1593161326994579) q[0];
ry(-0.8799545084831424) q[1];
cx q[0],q[1];
ry(1.6122796221606763) q[1];
ry(1.3336582307169662) q[2];
cx q[1],q[2];
ry(0.8830066751749239) q[1];
ry(0.8052064071062188) q[2];
cx q[1],q[2];
ry(1.3184150394390308) q[2];
ry(-1.364681915802107) q[3];
cx q[2],q[3];
ry(-2.7771651101263646) q[2];
ry(-1.6230800976561375) q[3];
cx q[2],q[3];
ry(-2.81357569750962) q[3];
ry(2.586189431176959) q[4];
cx q[3],q[4];
ry(0.45030524313595766) q[3];
ry(-1.3728402372831385) q[4];
cx q[3],q[4];
ry(1.0997943265364682) q[4];
ry(-1.4084416248950253) q[5];
cx q[4],q[5];
ry(-2.981938064926062) q[4];
ry(0.015612367661713478) q[5];
cx q[4],q[5];
ry(2.791704276346279) q[5];
ry(-1.1100306861409681) q[6];
cx q[5],q[6];
ry(1.978644752720248) q[5];
ry(-1.4428828505180857) q[6];
cx q[5],q[6];
ry(-0.08001988627257806) q[6];
ry(-2.694810459369775) q[7];
cx q[6],q[7];
ry(2.022588832902309) q[6];
ry(-0.9090117000613164) q[7];
cx q[6],q[7];
ry(-2.8132024791294046) q[0];
ry(-1.19009369985282) q[1];
cx q[0],q[1];
ry(-2.141956729331221) q[0];
ry(1.1942423892660183) q[1];
cx q[0],q[1];
ry(-1.0074873026766433) q[1];
ry(1.3939463269155081) q[2];
cx q[1],q[2];
ry(0.7740010811528064) q[1];
ry(0.7552003646884345) q[2];
cx q[1],q[2];
ry(0.32286348935828) q[2];
ry(-0.401011887154627) q[3];
cx q[2],q[3];
ry(2.0732652538030454) q[2];
ry(0.40324341691796894) q[3];
cx q[2],q[3];
ry(0.01786013791208596) q[3];
ry(1.253376476339602) q[4];
cx q[3],q[4];
ry(2.9537446800384215) q[3];
ry(-0.7850026762249422) q[4];
cx q[3],q[4];
ry(-1.0463503891638588) q[4];
ry(0.38416198909718346) q[5];
cx q[4],q[5];
ry(1.931913577858532) q[4];
ry(3.116626216856107) q[5];
cx q[4],q[5];
ry(-2.989047557584321) q[5];
ry(-0.9690343069966881) q[6];
cx q[5],q[6];
ry(0.7679022831823828) q[5];
ry(2.437283179052874) q[6];
cx q[5],q[6];
ry(-0.49010533812632673) q[6];
ry(1.1354187907247786) q[7];
cx q[6],q[7];
ry(1.3486247385949888) q[6];
ry(-1.1862434763678558) q[7];
cx q[6],q[7];
ry(-2.7008533265214973) q[0];
ry(1.9138412631568407) q[1];
cx q[0],q[1];
ry(-2.0080418135354163) q[0];
ry(-2.9853694275351033) q[1];
cx q[0],q[1];
ry(-0.07761434627244146) q[1];
ry(-1.588358573761439) q[2];
cx q[1],q[2];
ry(-2.063835496503885) q[1];
ry(0.9167740235653262) q[2];
cx q[1],q[2];
ry(1.1099379931546292) q[2];
ry(1.0643558460705207) q[3];
cx q[2],q[3];
ry(-0.5122980200746053) q[2];
ry(-1.6036347737558474) q[3];
cx q[2],q[3];
ry(-1.701174839668039) q[3];
ry(-0.47304175515641744) q[4];
cx q[3],q[4];
ry(-1.0493285702596458) q[3];
ry(1.0909594735087214) q[4];
cx q[3],q[4];
ry(1.8448317742917608) q[4];
ry(-0.474063152659439) q[5];
cx q[4],q[5];
ry(1.8736060719654473) q[4];
ry(1.8022088547751376) q[5];
cx q[4],q[5];
ry(-2.0091112766151706) q[5];
ry(1.2549687944825232) q[6];
cx q[5],q[6];
ry(-1.4728105032179286) q[5];
ry(-0.1371471368076265) q[6];
cx q[5],q[6];
ry(-0.5007366333383922) q[6];
ry(-2.2355043989169037) q[7];
cx q[6],q[7];
ry(1.743663898468844) q[6];
ry(-2.6249106575956302) q[7];
cx q[6],q[7];
ry(-0.3079482734842163) q[0];
ry(-1.4833966950529538) q[1];
cx q[0],q[1];
ry(-0.29265583500293074) q[0];
ry(1.970932942172666) q[1];
cx q[0],q[1];
ry(1.2295571435432837) q[1];
ry(-0.07808516608439753) q[2];
cx q[1],q[2];
ry(-2.1666937690840697) q[1];
ry(1.5841046680846613) q[2];
cx q[1],q[2];
ry(0.1689476827613209) q[2];
ry(-0.6048461046637539) q[3];
cx q[2],q[3];
ry(1.2184584992035479) q[2];
ry(1.2728466100668774) q[3];
cx q[2],q[3];
ry(-0.6266617231256832) q[3];
ry(1.1776563830668092) q[4];
cx q[3],q[4];
ry(-1.1294368455490047) q[3];
ry(-2.256639020298876) q[4];
cx q[3],q[4];
ry(-0.4391665486475121) q[4];
ry(2.965654786659014) q[5];
cx q[4],q[5];
ry(1.0420384644791576) q[4];
ry(-1.4352860457420504) q[5];
cx q[4],q[5];
ry(-0.6539148960622275) q[5];
ry(-1.748291009153629) q[6];
cx q[5],q[6];
ry(-2.338417761460221) q[5];
ry(-0.8508166458074854) q[6];
cx q[5],q[6];
ry(0.019627015156434204) q[6];
ry(2.314444437036022) q[7];
cx q[6],q[7];
ry(-2.484494052453619) q[6];
ry(2.7707988344678802) q[7];
cx q[6],q[7];
ry(-2.0396579993079236) q[0];
ry(-2.631088670893471) q[1];
cx q[0],q[1];
ry(-1.4973451124675665) q[0];
ry(1.355972663624086) q[1];
cx q[0],q[1];
ry(1.8843506732374262) q[1];
ry(1.499716594779114) q[2];
cx q[1],q[2];
ry(0.5217277695760268) q[1];
ry(-2.1207279161085544) q[2];
cx q[1],q[2];
ry(-2.4364545819285923) q[2];
ry(-2.0920600869465638) q[3];
cx q[2],q[3];
ry(0.4643584161946003) q[2];
ry(-0.39216172958997736) q[3];
cx q[2],q[3];
ry(-0.11839473318363501) q[3];
ry(1.003817839991961) q[4];
cx q[3],q[4];
ry(2.2877906097021112) q[3];
ry(-0.2143099197400488) q[4];
cx q[3],q[4];
ry(2.416177029591976) q[4];
ry(-2.0321581431191937) q[5];
cx q[4],q[5];
ry(-1.2110195069205192) q[4];
ry(-1.3874215490280681) q[5];
cx q[4],q[5];
ry(2.8956448372484815) q[5];
ry(1.1515175864463147) q[6];
cx q[5],q[6];
ry(1.161544682362923) q[5];
ry(1.6483440474946054) q[6];
cx q[5],q[6];
ry(-0.1001611970496894) q[6];
ry(-1.3497046312909609) q[7];
cx q[6],q[7];
ry(1.1169160344898676) q[6];
ry(1.4673439536164585) q[7];
cx q[6],q[7];
ry(1.937006891140151) q[0];
ry(0.0973517047160578) q[1];
cx q[0],q[1];
ry(-0.5274937636085234) q[0];
ry(0.30331202511851113) q[1];
cx q[0],q[1];
ry(0.5053554054723985) q[1];
ry(-1.756483612113332) q[2];
cx q[1],q[2];
ry(0.32768564180770726) q[1];
ry(2.234413939746071) q[2];
cx q[1],q[2];
ry(-2.656012016881935) q[2];
ry(-1.1649517115708985) q[3];
cx q[2],q[3];
ry(1.4782675682257944) q[2];
ry(-0.8226409285714303) q[3];
cx q[2],q[3];
ry(2.1913683911378996) q[3];
ry(-1.2985684566597118) q[4];
cx q[3],q[4];
ry(-1.240252040465833) q[3];
ry(-0.11754594036640853) q[4];
cx q[3],q[4];
ry(-2.1002252706376656) q[4];
ry(2.4450067084220866) q[5];
cx q[4],q[5];
ry(-0.6534371929065119) q[4];
ry(0.5002356299909102) q[5];
cx q[4],q[5];
ry(-2.8763667445516456) q[5];
ry(2.2556506626195265) q[6];
cx q[5],q[6];
ry(-1.4517365811026028) q[5];
ry(1.4900852895853163) q[6];
cx q[5],q[6];
ry(0.2683270877308817) q[6];
ry(2.396680909931085) q[7];
cx q[6],q[7];
ry(-0.19735961394760082) q[6];
ry(0.9061342809573469) q[7];
cx q[6],q[7];
ry(1.2512610604363168) q[0];
ry(-1.142750913795421) q[1];
cx q[0],q[1];
ry(-0.15329698383413876) q[0];
ry(1.921624891758242) q[1];
cx q[0],q[1];
ry(0.8730880293468399) q[1];
ry(-2.581019122845279) q[2];
cx q[1],q[2];
ry(-1.4508659180775245) q[1];
ry(-0.17902494958859094) q[2];
cx q[1],q[2];
ry(-0.3733247708838637) q[2];
ry(0.8367861359523342) q[3];
cx q[2],q[3];
ry(-1.111715595071118) q[2];
ry(-1.0150108769631574) q[3];
cx q[2],q[3];
ry(1.6162593998173396) q[3];
ry(-2.1680550456604086) q[4];
cx q[3],q[4];
ry(-0.19538580130184552) q[3];
ry(2.1649825354005516) q[4];
cx q[3],q[4];
ry(-2.14481780652872) q[4];
ry(-0.2889015436395169) q[5];
cx q[4],q[5];
ry(1.9216812919600463) q[4];
ry(-2.955825530001172) q[5];
cx q[4],q[5];
ry(1.3382807843503182) q[5];
ry(0.6805309607075294) q[6];
cx q[5],q[6];
ry(-1.5090666392696797) q[5];
ry(-1.079451262328937) q[6];
cx q[5],q[6];
ry(0.3024866075232385) q[6];
ry(3.1021791850651357) q[7];
cx q[6],q[7];
ry(0.6794668314673222) q[6];
ry(-1.1681866149707005) q[7];
cx q[6],q[7];
ry(1.2338179040174113) q[0];
ry(-1.6220464298653425) q[1];
cx q[0],q[1];
ry(1.8784802519535022) q[0];
ry(-0.9610551594114546) q[1];
cx q[0],q[1];
ry(0.9812509036511791) q[1];
ry(2.2394947527872135) q[2];
cx q[1],q[2];
ry(-0.0012243327896710454) q[1];
ry(0.42211654083182637) q[2];
cx q[1],q[2];
ry(-1.700028222315486) q[2];
ry(1.5362572435166504) q[3];
cx q[2],q[3];
ry(3.028925603774404) q[2];
ry(0.38375232933030556) q[3];
cx q[2],q[3];
ry(-2.1475582211933952) q[3];
ry(2.3275016522346244) q[4];
cx q[3],q[4];
ry(2.6900410117713114) q[3];
ry(1.284048367108965) q[4];
cx q[3],q[4];
ry(1.118025684166633) q[4];
ry(0.7347411231156112) q[5];
cx q[4],q[5];
ry(-2.6202270245173858) q[4];
ry(-2.29396086451325) q[5];
cx q[4],q[5];
ry(2.1548850211969715) q[5];
ry(0.585192153659901) q[6];
cx q[5],q[6];
ry(-0.444591233454027) q[5];
ry(-0.08814275592349424) q[6];
cx q[5],q[6];
ry(0.8506663302982842) q[6];
ry(-0.43263531769979746) q[7];
cx q[6],q[7];
ry(-0.02941422855674336) q[6];
ry(1.8148369251052578) q[7];
cx q[6],q[7];
ry(1.7430360263810778) q[0];
ry(-2.629237933572643) q[1];
cx q[0],q[1];
ry(-1.8533855736932976) q[0];
ry(1.010159839605703) q[1];
cx q[0],q[1];
ry(0.8492825005076511) q[1];
ry(-1.1046576666032077) q[2];
cx q[1],q[2];
ry(-1.2624308747753483) q[1];
ry(-2.6096763316875604) q[2];
cx q[1],q[2];
ry(-0.5132006415546783) q[2];
ry(-2.654443021190911) q[3];
cx q[2],q[3];
ry(-0.6289120919444029) q[2];
ry(2.506790609337546) q[3];
cx q[2],q[3];
ry(-2.9518196892374386) q[3];
ry(-2.694249003395549) q[4];
cx q[3],q[4];
ry(1.7521398030135265) q[3];
ry(0.19998149703634285) q[4];
cx q[3],q[4];
ry(2.942229568277486) q[4];
ry(-2.4864599170598813) q[5];
cx q[4],q[5];
ry(2.67233247942465) q[4];
ry(-0.4996155735016242) q[5];
cx q[4],q[5];
ry(0.31522582916933306) q[5];
ry(1.1130508949360083) q[6];
cx q[5],q[6];
ry(-2.7456943522158506) q[5];
ry(3.1154035685277206) q[6];
cx q[5],q[6];
ry(1.2330896590335438) q[6];
ry(-2.580031521499124) q[7];
cx q[6],q[7];
ry(0.44340137266798596) q[6];
ry(-2.3283908534368813) q[7];
cx q[6],q[7];
ry(-2.3665158671963074) q[0];
ry(-1.7681680002594715) q[1];
cx q[0],q[1];
ry(2.618400294009939) q[0];
ry(-0.003553203620400156) q[1];
cx q[0],q[1];
ry(-0.9504935601832073) q[1];
ry(-1.8752286210765243) q[2];
cx q[1],q[2];
ry(0.9027544722325171) q[1];
ry(1.1317464859793906) q[2];
cx q[1],q[2];
ry(-1.9614119664913692) q[2];
ry(-0.21614498654706485) q[3];
cx q[2],q[3];
ry(-1.6782113509964152) q[2];
ry(0.5666017340002617) q[3];
cx q[2],q[3];
ry(-2.258575883083936) q[3];
ry(2.270621166999885) q[4];
cx q[3],q[4];
ry(1.5155977155940537) q[3];
ry(0.7058209788950043) q[4];
cx q[3],q[4];
ry(0.3720811140657146) q[4];
ry(1.2415035675918014) q[5];
cx q[4],q[5];
ry(2.0028621615240416) q[4];
ry(2.9177673496912835) q[5];
cx q[4],q[5];
ry(3.0462972955824115) q[5];
ry(1.6603734810285413) q[6];
cx q[5],q[6];
ry(-0.10503990933542208) q[5];
ry(-2.6261516448154074) q[6];
cx q[5],q[6];
ry(-2.559342381705141) q[6];
ry(-0.17470402165957388) q[7];
cx q[6],q[7];
ry(-2.9737026937855493) q[6];
ry(2.723529423588726) q[7];
cx q[6],q[7];
ry(0.5085130590491085) q[0];
ry(1.8192718812041266) q[1];
cx q[0],q[1];
ry(0.5115990260325766) q[0];
ry(-0.20410330696982903) q[1];
cx q[0],q[1];
ry(-2.711287006213762) q[1];
ry(0.1493564312933974) q[2];
cx q[1],q[2];
ry(-2.2848700228992755) q[1];
ry(-1.185511118598705) q[2];
cx q[1],q[2];
ry(0.21778620500270218) q[2];
ry(-2.944990578063518) q[3];
cx q[2],q[3];
ry(-0.5806113814836874) q[2];
ry(0.20547170925863512) q[3];
cx q[2],q[3];
ry(1.2992773280694276) q[3];
ry(2.7230995404167824) q[4];
cx q[3],q[4];
ry(-0.05303020427144834) q[3];
ry(-2.5114826198474396) q[4];
cx q[3],q[4];
ry(-2.234654760254691) q[4];
ry(-0.3746237047858463) q[5];
cx q[4],q[5];
ry(-1.1509364680415866) q[4];
ry(-2.8465671176438514) q[5];
cx q[4],q[5];
ry(1.8250973519022207) q[5];
ry(2.0297895642025185) q[6];
cx q[5],q[6];
ry(0.3114568012898893) q[5];
ry(-0.48003223770849335) q[6];
cx q[5],q[6];
ry(-1.5512795291048151) q[6];
ry(1.3466932640515727) q[7];
cx q[6],q[7];
ry(-2.6536521644843227) q[6];
ry(2.5258925066997806) q[7];
cx q[6],q[7];
ry(0.9517780143714538) q[0];
ry(2.910428012392069) q[1];
cx q[0],q[1];
ry(0.20382898506382396) q[0];
ry(-0.42655919402490233) q[1];
cx q[0],q[1];
ry(2.103289513142477) q[1];
ry(1.9905659312536428) q[2];
cx q[1],q[2];
ry(-1.758963941912547) q[1];
ry(-1.2082766624307344) q[2];
cx q[1],q[2];
ry(2.397358500644214) q[2];
ry(-1.6325381428887304) q[3];
cx q[2],q[3];
ry(1.7266832686486504) q[2];
ry(0.00916736641949614) q[3];
cx q[2],q[3];
ry(-2.752634315288268) q[3];
ry(1.6627437515087038) q[4];
cx q[3],q[4];
ry(2.065200795841225) q[3];
ry(0.6571852721277629) q[4];
cx q[3],q[4];
ry(-1.5929695538751396) q[4];
ry(2.8263274366288886) q[5];
cx q[4],q[5];
ry(0.7847902623463305) q[4];
ry(3.104312743669807) q[5];
cx q[4],q[5];
ry(-1.598819724384687) q[5];
ry(-1.7884997839310555) q[6];
cx q[5],q[6];
ry(-2.419899836299227) q[5];
ry(-0.365494452293879) q[6];
cx q[5],q[6];
ry(-1.2510562119647293) q[6];
ry(-1.1718032721530622) q[7];
cx q[6],q[7];
ry(1.3317530154267023) q[6];
ry(-0.5727189160950568) q[7];
cx q[6],q[7];
ry(0.7449881444396554) q[0];
ry(-2.295555685042824) q[1];
cx q[0],q[1];
ry(0.2242470014416423) q[0];
ry(-2.503944001382401) q[1];
cx q[0],q[1];
ry(-1.733720679944824) q[1];
ry(-1.710796740908812) q[2];
cx q[1],q[2];
ry(-0.8241207935899055) q[1];
ry(1.40647267153693) q[2];
cx q[1],q[2];
ry(-2.3685606425881995) q[2];
ry(1.5521388800379519) q[3];
cx q[2],q[3];
ry(-0.8579097644986018) q[2];
ry(2.4856043210658614) q[3];
cx q[2],q[3];
ry(0.2738527187541182) q[3];
ry(-2.0075773460967055) q[4];
cx q[3],q[4];
ry(0.7561654893084979) q[3];
ry(-3.100699378129068) q[4];
cx q[3],q[4];
ry(-0.9713906924298195) q[4];
ry(1.3981365250878302) q[5];
cx q[4],q[5];
ry(1.3834886131128874) q[4];
ry(0.8865656451944826) q[5];
cx q[4],q[5];
ry(2.0936372417401916) q[5];
ry(1.178862788878879) q[6];
cx q[5],q[6];
ry(1.7822793138325403) q[5];
ry(1.3682136313438376) q[6];
cx q[5],q[6];
ry(1.1890789898195897) q[6];
ry(-1.0616848490018835) q[7];
cx q[6],q[7];
ry(-2.3584034281733475) q[6];
ry(2.8534886166812274) q[7];
cx q[6],q[7];
ry(-0.8244366475814948) q[0];
ry(-0.33443877730497096) q[1];
cx q[0],q[1];
ry(0.7559867580211653) q[0];
ry(-1.1213491441455528) q[1];
cx q[0],q[1];
ry(-1.6877207809708388) q[1];
ry(0.9256551991468926) q[2];
cx q[1],q[2];
ry(-1.7731989131360786) q[1];
ry(1.4329590984447718) q[2];
cx q[1],q[2];
ry(-3.126254588882888) q[2];
ry(0.380672408995352) q[3];
cx q[2],q[3];
ry(-0.31196170721106287) q[2];
ry(0.5601923523402851) q[3];
cx q[2],q[3];
ry(-2.1693462909900756) q[3];
ry(-0.8335088573822063) q[4];
cx q[3],q[4];
ry(1.5000011774228292) q[3];
ry(-0.3626229065169344) q[4];
cx q[3],q[4];
ry(-0.8478393802075797) q[4];
ry(2.740268595350231) q[5];
cx q[4],q[5];
ry(-2.1261631347488508) q[4];
ry(-1.589156023318436) q[5];
cx q[4],q[5];
ry(-2.676025479108969) q[5];
ry(-0.3822510089192738) q[6];
cx q[5],q[6];
ry(1.4722331809945157) q[5];
ry(2.470029341645257) q[6];
cx q[5],q[6];
ry(1.5831535089491984) q[6];
ry(-0.8168805665141283) q[7];
cx q[6],q[7];
ry(0.6758326360238689) q[6];
ry(-0.6100151052989498) q[7];
cx q[6],q[7];
ry(1.2634345335440775) q[0];
ry(-2.061700997291454) q[1];
cx q[0],q[1];
ry(1.323188109913998) q[0];
ry(1.2107332793830183) q[1];
cx q[0],q[1];
ry(-1.872253419281649) q[1];
ry(0.9170436721610258) q[2];
cx q[1],q[2];
ry(-2.050144254918414) q[1];
ry(1.771962015974183) q[2];
cx q[1],q[2];
ry(0.5346143188947616) q[2];
ry(-0.3369486581141005) q[3];
cx q[2],q[3];
ry(-1.4512741171247896) q[2];
ry(1.0899624205445406) q[3];
cx q[2],q[3];
ry(2.5668319792199163) q[3];
ry(-0.5257134959581995) q[4];
cx q[3],q[4];
ry(2.6605971356624285) q[3];
ry(-2.8827578015556314) q[4];
cx q[3],q[4];
ry(0.5209176647787322) q[4];
ry(-2.4232464583046855) q[5];
cx q[4],q[5];
ry(-2.1366147313660413) q[4];
ry(-0.009560954427015973) q[5];
cx q[4],q[5];
ry(1.4193901296618154) q[5];
ry(1.2880589710084345) q[6];
cx q[5],q[6];
ry(1.9098462673420866) q[5];
ry(-1.369001023223071) q[6];
cx q[5],q[6];
ry(-0.25281627875942814) q[6];
ry(2.2268234654157224) q[7];
cx q[6],q[7];
ry(1.996282282042288) q[6];
ry(-1.8562905433919807) q[7];
cx q[6],q[7];
ry(0.7194411543956125) q[0];
ry(-0.7661681808716372) q[1];
cx q[0],q[1];
ry(2.748377964578254) q[0];
ry(-2.9562141083849602) q[1];
cx q[0],q[1];
ry(-0.4972914165373403) q[1];
ry(1.3587372944280642) q[2];
cx q[1],q[2];
ry(-0.4924734508731027) q[1];
ry(0.49178567271076457) q[2];
cx q[1],q[2];
ry(1.1896587604714874) q[2];
ry(0.9803980314117098) q[3];
cx q[2],q[3];
ry(0.9722304969291578) q[2];
ry(0.30612230158093096) q[3];
cx q[2],q[3];
ry(-0.8402612898093897) q[3];
ry(-2.8531824401970582) q[4];
cx q[3],q[4];
ry(0.05164787803949267) q[3];
ry(1.191525071680302) q[4];
cx q[3],q[4];
ry(-2.579532429202983) q[4];
ry(-1.6337389940224991) q[5];
cx q[4],q[5];
ry(1.4986471399465788) q[4];
ry(2.9211216448763904) q[5];
cx q[4],q[5];
ry(-0.7838458299321981) q[5];
ry(2.3089203774628877) q[6];
cx q[5],q[6];
ry(0.36482467235348776) q[5];
ry(-0.6017101983114269) q[6];
cx q[5],q[6];
ry(-2.361615998835707) q[6];
ry(2.9047892439682674) q[7];
cx q[6],q[7];
ry(-2.381589292223351) q[6];
ry(1.7386731405916205) q[7];
cx q[6],q[7];
ry(-2.704579336178827) q[0];
ry(0.12886592673237232) q[1];
cx q[0],q[1];
ry(-1.3075122649564797) q[0];
ry(-0.7152842982322528) q[1];
cx q[0],q[1];
ry(2.730322322347978) q[1];
ry(2.4391905854570455) q[2];
cx q[1],q[2];
ry(1.1649055378793491) q[1];
ry(-2.8580344817868943) q[2];
cx q[1],q[2];
ry(-0.36520317273971115) q[2];
ry(0.9875969303550454) q[3];
cx q[2],q[3];
ry(-0.7289462417345005) q[2];
ry(-2.4085178720498934) q[3];
cx q[2],q[3];
ry(1.196547101064355) q[3];
ry(2.969263142851368) q[4];
cx q[3],q[4];
ry(-3.0088541195524066) q[3];
ry(-2.424928116863112) q[4];
cx q[3],q[4];
ry(-2.4955797741330934) q[4];
ry(0.9190867206013777) q[5];
cx q[4],q[5];
ry(-2.763303891371041) q[4];
ry(-2.2887934058983563) q[5];
cx q[4],q[5];
ry(1.8493602101845783) q[5];
ry(2.3005436430628947) q[6];
cx q[5],q[6];
ry(-3.111257903004098) q[5];
ry(-2.820659392048178) q[6];
cx q[5],q[6];
ry(-2.0093516637930815) q[6];
ry(0.39729622873110687) q[7];
cx q[6],q[7];
ry(0.7445792222125878) q[6];
ry(-1.0032757560393941) q[7];
cx q[6],q[7];
ry(-2.283562565855137) q[0];
ry(-1.7277566858996458) q[1];
cx q[0],q[1];
ry(2.352662995193106) q[0];
ry(-2.1659360140101187) q[1];
cx q[0],q[1];
ry(0.9192587570338137) q[1];
ry(0.9416688829542004) q[2];
cx q[1],q[2];
ry(-0.9269237119749355) q[1];
ry(2.782564571996315) q[2];
cx q[1],q[2];
ry(-3.0640453222444166) q[2];
ry(1.0632103991602646) q[3];
cx q[2],q[3];
ry(-3.0077396522152546) q[2];
ry(1.7640679748927308) q[3];
cx q[2],q[3];
ry(-2.208431903778779) q[3];
ry(-3.053848668913553) q[4];
cx q[3],q[4];
ry(2.7882707760891927) q[3];
ry(-2.241984644297135) q[4];
cx q[3],q[4];
ry(2.3199624305360302) q[4];
ry(1.1680413428850596) q[5];
cx q[4],q[5];
ry(-2.3717658767368666) q[4];
ry(1.5879308487115147) q[5];
cx q[4],q[5];
ry(-1.4918538963961625) q[5];
ry(0.13641955633804284) q[6];
cx q[5],q[6];
ry(-2.1425590488744186) q[5];
ry(-2.7317526378750743) q[6];
cx q[5],q[6];
ry(-0.8932392032635538) q[6];
ry(2.6258884050430966) q[7];
cx q[6],q[7];
ry(-0.4178394069984419) q[6];
ry(1.6790773783355766) q[7];
cx q[6],q[7];
ry(-3.0890937884289644) q[0];
ry(2.1434654234655586) q[1];
cx q[0],q[1];
ry(1.9261790755031207) q[0];
ry(-0.6634462347436562) q[1];
cx q[0],q[1];
ry(0.41756447104105665) q[1];
ry(2.4233079871334287) q[2];
cx q[1],q[2];
ry(1.0112898447953416) q[1];
ry(1.5966248443450215) q[2];
cx q[1],q[2];
ry(0.7669479570548668) q[2];
ry(1.1779735033545267) q[3];
cx q[2],q[3];
ry(-1.4471531125060597) q[2];
ry(-1.0380518175079385) q[3];
cx q[2],q[3];
ry(-3.035545812963836) q[3];
ry(2.767716909941974) q[4];
cx q[3],q[4];
ry(-2.5415222980533305) q[3];
ry(0.3821009406886331) q[4];
cx q[3],q[4];
ry(1.496239468070871) q[4];
ry(-3.033558300889285) q[5];
cx q[4],q[5];
ry(-0.8966448467296262) q[4];
ry(-0.2898790366270054) q[5];
cx q[4],q[5];
ry(-2.8042216470421626) q[5];
ry(0.9366379989035973) q[6];
cx q[5],q[6];
ry(3.057227857022522) q[5];
ry(-0.11203994070015058) q[6];
cx q[5],q[6];
ry(-2.781505308737344) q[6];
ry(-1.701649111292619) q[7];
cx q[6],q[7];
ry(2.9074160135265963) q[6];
ry(0.8526113252931921) q[7];
cx q[6],q[7];
ry(2.6581056401566254) q[0];
ry(2.1798825086155365) q[1];
cx q[0],q[1];
ry(1.7702492449193248) q[0];
ry(-1.8913121338230963) q[1];
cx q[0],q[1];
ry(-2.46975452136326) q[1];
ry(2.0454698836947776) q[2];
cx q[1],q[2];
ry(-2.4528961099883033) q[1];
ry(-0.2143009541979248) q[2];
cx q[1],q[2];
ry(2.989114217026408) q[2];
ry(-1.6247327704273005) q[3];
cx q[2],q[3];
ry(-1.9023332790384773) q[2];
ry(0.8283494620454945) q[3];
cx q[2],q[3];
ry(-1.9381318875163487) q[3];
ry(-1.2935587888445275) q[4];
cx q[3],q[4];
ry(0.0378990756218894) q[3];
ry(2.146179189223118) q[4];
cx q[3],q[4];
ry(3.0338871462297283) q[4];
ry(-1.214167938726973) q[5];
cx q[4],q[5];
ry(-2.153209147247007) q[4];
ry(2.899206673049116) q[5];
cx q[4],q[5];
ry(1.6814194126094817) q[5];
ry(2.740833503422063) q[6];
cx q[5],q[6];
ry(2.584702855748881) q[5];
ry(1.138249185080034) q[6];
cx q[5],q[6];
ry(-2.101290165285473) q[6];
ry(2.499812052449064) q[7];
cx q[6],q[7];
ry(2.2117088904712157) q[6];
ry(-1.5045649374934156) q[7];
cx q[6],q[7];
ry(-0.06653598624894652) q[0];
ry(1.4483670553747698) q[1];
cx q[0],q[1];
ry(-1.736755373316421) q[0];
ry(2.3430041303723943) q[1];
cx q[0],q[1];
ry(1.9939700388532433) q[1];
ry(1.9924897036810298) q[2];
cx q[1],q[2];
ry(2.7931180784025544) q[1];
ry(1.7998958751741982) q[2];
cx q[1],q[2];
ry(-0.1868317395430087) q[2];
ry(-0.5081690686908785) q[3];
cx q[2],q[3];
ry(-0.30353024113588756) q[2];
ry(-2.858275003522014) q[3];
cx q[2],q[3];
ry(0.2909470985165221) q[3];
ry(1.0668383299985003) q[4];
cx q[3],q[4];
ry(-1.2763752648823434) q[3];
ry(-0.1494144070764634) q[4];
cx q[3],q[4];
ry(2.299104214383981) q[4];
ry(2.745998459483253) q[5];
cx q[4],q[5];
ry(-2.0070475441919307) q[4];
ry(-1.9760409183813723) q[5];
cx q[4],q[5];
ry(1.1669895179956618) q[5];
ry(-1.6812425493653382) q[6];
cx q[5],q[6];
ry(2.5676186645548036) q[5];
ry(2.472862186584734) q[6];
cx q[5],q[6];
ry(0.7337562713993806) q[6];
ry(0.7266336814378141) q[7];
cx q[6],q[7];
ry(-0.373259454550376) q[6];
ry(0.7375887018425491) q[7];
cx q[6],q[7];
ry(2.7349956614989335) q[0];
ry(-0.12043817736551521) q[1];
cx q[0],q[1];
ry(-2.8126883054360015) q[0];
ry(-0.2343808664188085) q[1];
cx q[0],q[1];
ry(-2.2449896654319077) q[1];
ry(2.4919096576850595) q[2];
cx q[1],q[2];
ry(3.0236056464378893) q[1];
ry(-0.28368275543828947) q[2];
cx q[1],q[2];
ry(0.40597653954887836) q[2];
ry(-0.5910519505940154) q[3];
cx q[2],q[3];
ry(2.733445152660354) q[2];
ry(2.780261209268322) q[3];
cx q[2],q[3];
ry(-0.42649452265866655) q[3];
ry(1.2337898722610072) q[4];
cx q[3],q[4];
ry(0.46312900731344797) q[3];
ry(-1.8919448364011313) q[4];
cx q[3],q[4];
ry(1.1802641936374942) q[4];
ry(-0.5464873789465552) q[5];
cx q[4],q[5];
ry(-1.5825650327091587) q[4];
ry(-0.2190636430514079) q[5];
cx q[4],q[5];
ry(-1.8095697806095208) q[5];
ry(-2.214147785882947) q[6];
cx q[5],q[6];
ry(0.9468884221059195) q[5];
ry(2.491953591359704) q[6];
cx q[5],q[6];
ry(1.2961686065606057) q[6];
ry(2.93005149716764) q[7];
cx q[6],q[7];
ry(1.3941082994949174) q[6];
ry(-1.8495584021663438) q[7];
cx q[6],q[7];
ry(2.1564591412548686) q[0];
ry(2.987671509183415) q[1];
cx q[0],q[1];
ry(-2.4511391167212193) q[0];
ry(0.5022413686353255) q[1];
cx q[0],q[1];
ry(1.358398347689426) q[1];
ry(3.0964909135421186) q[2];
cx q[1],q[2];
ry(-2.434868723130545) q[1];
ry(2.1692380463613343) q[2];
cx q[1],q[2];
ry(0.33760032349894775) q[2];
ry(-0.672665268832428) q[3];
cx q[2],q[3];
ry(1.769255462249486) q[2];
ry(1.325949864513609) q[3];
cx q[2],q[3];
ry(0.08508355068856854) q[3];
ry(-1.7371637238102942) q[4];
cx q[3],q[4];
ry(2.4542611400767056) q[3];
ry(2.7000860091431997) q[4];
cx q[3],q[4];
ry(-1.2535762271105675) q[4];
ry(0.5153320061074149) q[5];
cx q[4],q[5];
ry(3.0081411391981505) q[4];
ry(1.524684514351617) q[5];
cx q[4],q[5];
ry(0.1285101698320128) q[5];
ry(0.2982545494475608) q[6];
cx q[5],q[6];
ry(0.8228311021783137) q[5];
ry(-1.5729659552064152) q[6];
cx q[5],q[6];
ry(-2.713579245943491) q[6];
ry(-0.9668459915600129) q[7];
cx q[6],q[7];
ry(3.0029573971561465) q[6];
ry(-1.4774986615517935) q[7];
cx q[6],q[7];
ry(0.31362289710112906) q[0];
ry(-2.249467055895612) q[1];
cx q[0],q[1];
ry(0.9358263131810158) q[0];
ry(0.5521257894502184) q[1];
cx q[0],q[1];
ry(1.716827253090017) q[1];
ry(-3.110875192078042) q[2];
cx q[1],q[2];
ry(2.5829750170284664) q[1];
ry(3.0190048741371296) q[2];
cx q[1],q[2];
ry(1.2953888928576944) q[2];
ry(-2.4661406976937066) q[3];
cx q[2],q[3];
ry(2.8716265107953713) q[2];
ry(1.35909412837271) q[3];
cx q[2],q[3];
ry(0.0037527045967402565) q[3];
ry(-1.415252425319468) q[4];
cx q[3],q[4];
ry(1.0765339874580109) q[3];
ry(0.4491399214221694) q[4];
cx q[3],q[4];
ry(3.041323304535515) q[4];
ry(-0.20548120217392984) q[5];
cx q[4],q[5];
ry(-1.7395948756956279) q[4];
ry(-2.850263240083897) q[5];
cx q[4],q[5];
ry(2.182352010259895) q[5];
ry(1.479798693260497) q[6];
cx q[5],q[6];
ry(0.0035436909339402294) q[5];
ry(0.656326819838526) q[6];
cx q[5],q[6];
ry(-1.594101807678258) q[6];
ry(-1.8130400851988562) q[7];
cx q[6],q[7];
ry(3.008731542499966) q[6];
ry(3.0213917253001012) q[7];
cx q[6],q[7];
ry(1.635610147032852) q[0];
ry(-0.5531614877795263) q[1];
cx q[0],q[1];
ry(0.30007063153833263) q[0];
ry(-1.8349401940785945) q[1];
cx q[0],q[1];
ry(-0.17731025612164422) q[1];
ry(-2.3567662881992266) q[2];
cx q[1],q[2];
ry(0.5242999589752259) q[1];
ry(-3.0953043191018472) q[2];
cx q[1],q[2];
ry(1.153349269302808) q[2];
ry(-3.0653897294553363) q[3];
cx q[2],q[3];
ry(2.8750351470346214) q[2];
ry(1.2802897090645136) q[3];
cx q[2],q[3];
ry(1.9289357349746128) q[3];
ry(0.6150376707836478) q[4];
cx q[3],q[4];
ry(1.640382329457589) q[3];
ry(1.4422142109286735) q[4];
cx q[3],q[4];
ry(-1.9206596317392481) q[4];
ry(0.5043318493916944) q[5];
cx q[4],q[5];
ry(1.0939415600734623) q[4];
ry(-0.2848931356994342) q[5];
cx q[4],q[5];
ry(1.287804133984122) q[5];
ry(1.7100819577559105) q[6];
cx q[5],q[6];
ry(3.13652934834809) q[5];
ry(-0.052982610725843315) q[6];
cx q[5],q[6];
ry(-0.48050911352801273) q[6];
ry(1.97225290662667) q[7];
cx q[6],q[7];
ry(-1.4631648674209876) q[6];
ry(-2.861515532933169) q[7];
cx q[6],q[7];
ry(3.034312121023649) q[0];
ry(-0.6774420659986743) q[1];
cx q[0],q[1];
ry(-2.9483035009929712) q[0];
ry(0.2088845664203127) q[1];
cx q[0],q[1];
ry(2.149481525864034) q[1];
ry(-2.846263748643553) q[2];
cx q[1],q[2];
ry(1.1390640820620062) q[1];
ry(2.365794409404495) q[2];
cx q[1],q[2];
ry(1.661493156755561) q[2];
ry(1.1254806333412635) q[3];
cx q[2],q[3];
ry(0.5492752143203197) q[2];
ry(2.542050735491491) q[3];
cx q[2],q[3];
ry(-0.03224945385275113) q[3];
ry(2.0436417080014535) q[4];
cx q[3],q[4];
ry(2.95555846329727) q[3];
ry(-1.810315329385915) q[4];
cx q[3],q[4];
ry(1.4217110174161416) q[4];
ry(1.0541785494634794) q[5];
cx q[4],q[5];
ry(-1.4076405644453172) q[4];
ry(2.2849345623731465) q[5];
cx q[4],q[5];
ry(2.4150624697263963) q[5];
ry(-2.8336312458236157) q[6];
cx q[5],q[6];
ry(0.2218799439117346) q[5];
ry(1.0397337843135381) q[6];
cx q[5],q[6];
ry(-0.7854646710655826) q[6];
ry(-2.271902250151796) q[7];
cx q[6],q[7];
ry(-1.5533388920555167) q[6];
ry(-3.084311591991552) q[7];
cx q[6],q[7];
ry(1.7832100147796983) q[0];
ry(-2.621293543898223) q[1];
ry(-0.6244113796771681) q[2];
ry(-1.403817489587662) q[3];
ry(-1.6893037329885736) q[4];
ry(-1.1255605262418973) q[5];
ry(1.8170687331441489) q[6];
ry(-1.1972758153708618) q[7];