OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.9777021491962054) q[0];
ry(-2.765674110934071) q[1];
cx q[0],q[1];
ry(-1.4316869215276435) q[0];
ry(-1.4412094255186303) q[1];
cx q[0],q[1];
ry(-2.0751633876320064) q[1];
ry(0.5216678416731712) q[2];
cx q[1],q[2];
ry(2.8379575854609445) q[1];
ry(-3.128641901423633) q[2];
cx q[1],q[2];
ry(1.9523486557945433) q[2];
ry(1.72282093179665) q[3];
cx q[2],q[3];
ry(1.7271002027228395) q[2];
ry(2.0917453222674305) q[3];
cx q[2],q[3];
ry(2.7280564096848403) q[3];
ry(-0.6005687009081155) q[4];
cx q[3],q[4];
ry(1.2012082783953364) q[3];
ry(0.1912477719106711) q[4];
cx q[3],q[4];
ry(-2.581064579293787) q[4];
ry(2.9351498106083325) q[5];
cx q[4],q[5];
ry(-0.015597959332496749) q[4];
ry(-3.125984107142473) q[5];
cx q[4],q[5];
ry(2.0853863832485207) q[5];
ry(-1.5996051574755608) q[6];
cx q[5],q[6];
ry(2.9010611557470165) q[5];
ry(1.7143249962819784) q[6];
cx q[5],q[6];
ry(-2.504734455012561) q[6];
ry(-1.575732962652354) q[7];
cx q[6],q[7];
ry(0.21369722256054313) q[6];
ry(3.1275625015801594) q[7];
cx q[6],q[7];
ry(-2.478558176684135) q[7];
ry(1.234872817164221) q[8];
cx q[7],q[8];
ry(-1.8133347728378197) q[7];
ry(1.778392705532692) q[8];
cx q[7],q[8];
ry(-1.9373219137804343) q[8];
ry(-2.2046124219939824) q[9];
cx q[8],q[9];
ry(-3.1394137184583846) q[8];
ry(3.138069033055814) q[9];
cx q[8],q[9];
ry(-2.7839059501798764) q[9];
ry(-0.7175689403674417) q[10];
cx q[9],q[10];
ry(0.012399221887379061) q[9];
ry(1.0808067173625648) q[10];
cx q[9],q[10];
ry(0.4377091786337168) q[10];
ry(0.1551634728949585) q[11];
cx q[10],q[11];
ry(-0.6068369229324801) q[10];
ry(-1.7733779011896003) q[11];
cx q[10],q[11];
ry(1.071064354483176) q[0];
ry(2.026295743899155) q[1];
cx q[0],q[1];
ry(1.4737643285907822) q[0];
ry(-1.2460913377831186) q[1];
cx q[0],q[1];
ry(2.8071138164403564) q[1];
ry(0.19962486723979644) q[2];
cx q[1],q[2];
ry(2.3725432244731044) q[1];
ry(3.122579305758968) q[2];
cx q[1],q[2];
ry(-2.2426177746638922) q[2];
ry(-1.9522152023341952) q[3];
cx q[2],q[3];
ry(2.8353136020424805) q[2];
ry(-1.327053381815357) q[3];
cx q[2],q[3];
ry(0.1811203861710678) q[3];
ry(1.384020609730312) q[4];
cx q[3],q[4];
ry(0.72543408784838) q[3];
ry(-0.979640571690366) q[4];
cx q[3],q[4];
ry(0.5316483356238956) q[4];
ry(0.0468146233480713) q[5];
cx q[4],q[5];
ry(-2.0454088563690522) q[4];
ry(0.15262852590789588) q[5];
cx q[4],q[5];
ry(-2.82533551901773) q[5];
ry(-2.1398253224829484) q[6];
cx q[5],q[6];
ry(-0.34454417442661445) q[5];
ry(-1.0394813831060459) q[6];
cx q[5],q[6];
ry(-1.6876006275053619) q[6];
ry(-0.7739236048946951) q[7];
cx q[6],q[7];
ry(2.8077030795113416) q[6];
ry(-1.4451888473292003) q[7];
cx q[6],q[7];
ry(-0.10476812342022068) q[7];
ry(0.25310412817645794) q[8];
cx q[7],q[8];
ry(-1.631088901959072) q[7];
ry(1.5122392398655673) q[8];
cx q[7],q[8];
ry(-0.0957454345033319) q[8];
ry(3.043385242378769) q[9];
cx q[8],q[9];
ry(-0.06101873483404763) q[8];
ry(-2.8815672195093405) q[9];
cx q[8],q[9];
ry(2.648486887710795) q[9];
ry(0.7707078243810814) q[10];
cx q[9],q[10];
ry(0.9494204802345632) q[9];
ry(-1.0897936024900625) q[10];
cx q[9],q[10];
ry(2.447947333442546) q[10];
ry(0.47077906409224557) q[11];
cx q[10],q[11];
ry(-2.2208136307872213) q[10];
ry(-0.10792431237144484) q[11];
cx q[10],q[11];
ry(-1.9090597236784328) q[0];
ry(1.9639793859223371) q[1];
cx q[0],q[1];
ry(-0.25469828070727923) q[0];
ry(-0.9884337402532415) q[1];
cx q[0],q[1];
ry(0.024706673946009845) q[1];
ry(-0.9627894599337292) q[2];
cx q[1],q[2];
ry(1.5416899170609788) q[1];
ry(-3.0457335124020677) q[2];
cx q[1],q[2];
ry(2.9717549038595967) q[2];
ry(-2.4216338529041717) q[3];
cx q[2],q[3];
ry(-0.9651780885223413) q[2];
ry(0.6168081678368975) q[3];
cx q[2],q[3];
ry(0.3199107488978551) q[3];
ry(1.4479472275817207) q[4];
cx q[3],q[4];
ry(2.5102899116005752) q[3];
ry(-2.9451140143509136) q[4];
cx q[3],q[4];
ry(1.4444900472574862) q[4];
ry(2.2751014973640853) q[5];
cx q[4],q[5];
ry(-0.34649297947232505) q[4];
ry(3.0827839853800434) q[5];
cx q[4],q[5];
ry(-3.03703537060408) q[5];
ry(-3.1283765901406957) q[6];
cx q[5],q[6];
ry(1.3253199322227347) q[5];
ry(1.5160810245861667) q[6];
cx q[5],q[6];
ry(-2.4758491896975405) q[6];
ry(-1.9779383809201385) q[7];
cx q[6],q[7];
ry(-2.918395155477161) q[6];
ry(0.03194322802205996) q[7];
cx q[6],q[7];
ry(-3.1307313431634562) q[7];
ry(-0.6428776066640582) q[8];
cx q[7],q[8];
ry(-2.6489742993187013) q[7];
ry(-3.0586026990698123) q[8];
cx q[7],q[8];
ry(0.27382665441640963) q[8];
ry(-3.0680814765324405) q[9];
cx q[8],q[9];
ry(0.0679505191889751) q[8];
ry(2.53988760746502) q[9];
cx q[8],q[9];
ry(2.333833022099821) q[9];
ry(-0.13919171973700276) q[10];
cx q[9],q[10];
ry(-0.5939014371408087) q[9];
ry(0.1571149071902356) q[10];
cx q[9],q[10];
ry(2.1602583241447557) q[10];
ry(-0.4098365499046017) q[11];
cx q[10],q[11];
ry(3.0772563685710064) q[10];
ry(-1.5653911862042742) q[11];
cx q[10],q[11];
ry(0.5941414500564941) q[0];
ry(1.1767221339116634) q[1];
cx q[0],q[1];
ry(-2.846620636207662) q[0];
ry(2.8327681188928975) q[1];
cx q[0],q[1];
ry(-1.3553262845465432) q[1];
ry(1.3269108224291877) q[2];
cx q[1],q[2];
ry(-0.9586976360702336) q[1];
ry(0.05836628393735399) q[2];
cx q[1],q[2];
ry(-0.2807462411166881) q[2];
ry(3.0371599045776767) q[3];
cx q[2],q[3];
ry(0.030134204555880173) q[2];
ry(0.1545270502116951) q[3];
cx q[2],q[3];
ry(-2.1801734221103555) q[3];
ry(-1.7129807674093647) q[4];
cx q[3],q[4];
ry(-2.073349724794867) q[3];
ry(-0.8638049217897795) q[4];
cx q[3],q[4];
ry(-1.788373698778277) q[4];
ry(0.5218327644741149) q[5];
cx q[4],q[5];
ry(-0.30324791441951054) q[4];
ry(0.06509721960978752) q[5];
cx q[4],q[5];
ry(-0.43257950916989696) q[5];
ry(2.4961711526022325) q[6];
cx q[5],q[6];
ry(1.7385980633339075) q[5];
ry(0.8057706062370548) q[6];
cx q[5],q[6];
ry(2.6016966758999702) q[6];
ry(-0.4607984143172352) q[7];
cx q[6],q[7];
ry(3.105621827031523) q[6];
ry(0.04084313210260371) q[7];
cx q[6],q[7];
ry(-2.3309427396966207) q[7];
ry(1.243781863629219) q[8];
cx q[7],q[8];
ry(-0.8670191386436068) q[7];
ry(-0.009209736030506323) q[8];
cx q[7],q[8];
ry(0.9544327731222871) q[8];
ry(-2.2250900021823705) q[9];
cx q[8],q[9];
ry(-1.1264477881667503) q[8];
ry(-1.6894762584487264) q[9];
cx q[8],q[9];
ry(-2.9063081766737224) q[9];
ry(-1.3528430521129673) q[10];
cx q[9],q[10];
ry(2.9051546768488525) q[9];
ry(-0.08930465571011846) q[10];
cx q[9],q[10];
ry(-2.738089911797671) q[10];
ry(-3.034575126746762) q[11];
cx q[10],q[11];
ry(3.127854794423015) q[10];
ry(-0.46935295846913916) q[11];
cx q[10],q[11];
ry(2.576838170131394) q[0];
ry(-1.0523920984003954) q[1];
cx q[0],q[1];
ry(-1.6834195786833996) q[0];
ry(-2.577921906736872) q[1];
cx q[0],q[1];
ry(2.244290383135442) q[1];
ry(2.5584500634132294) q[2];
cx q[1],q[2];
ry(-0.09148749133783163) q[1];
ry(-3.0656113865555366) q[2];
cx q[1],q[2];
ry(-0.08251428897019236) q[2];
ry(2.0263776500856165) q[3];
cx q[2],q[3];
ry(0.021485105149825934) q[2];
ry(1.0427773436143504) q[3];
cx q[2],q[3];
ry(0.6186316674933181) q[3];
ry(-0.816019520288679) q[4];
cx q[3],q[4];
ry(-2.760239544845699) q[3];
ry(2.066885165528195) q[4];
cx q[3],q[4];
ry(0.588238555768809) q[4];
ry(1.9673811993380494) q[5];
cx q[4],q[5];
ry(2.872882719354613) q[4];
ry(2.9634457926830255) q[5];
cx q[4],q[5];
ry(-0.6804203242088347) q[5];
ry(0.8260399468677289) q[6];
cx q[5],q[6];
ry(-2.2323622921194444) q[5];
ry(1.8234215967391614) q[6];
cx q[5],q[6];
ry(-1.2287403611708791) q[6];
ry(1.9562000530114718) q[7];
cx q[6],q[7];
ry(-0.02907224965191721) q[6];
ry(1.5906764723590987) q[7];
cx q[6],q[7];
ry(-1.730114042768501) q[7];
ry(-2.096620607773403) q[8];
cx q[7],q[8];
ry(0.31382953945674963) q[7];
ry(-0.11144623541277965) q[8];
cx q[7],q[8];
ry(-0.6318959820368528) q[8];
ry(-2.091695919550507) q[9];
cx q[8],q[9];
ry(-2.662281770578173) q[8];
ry(-3.0947725948747657) q[9];
cx q[8],q[9];
ry(-1.5601939377430503) q[9];
ry(-0.38816062890200825) q[10];
cx q[9],q[10];
ry(-0.04754162316572282) q[9];
ry(0.004483040915249425) q[10];
cx q[9],q[10];
ry(0.6235086288552073) q[10];
ry(-1.3241515875836276) q[11];
cx q[10],q[11];
ry(0.6339034628675362) q[10];
ry(-2.57095021377277) q[11];
cx q[10],q[11];
ry(-0.7945893642885213) q[0];
ry(-2.987748153429616) q[1];
cx q[0],q[1];
ry(3.0208389203257493) q[0];
ry(-1.4223187634751218) q[1];
cx q[0],q[1];
ry(-0.25240771476985735) q[1];
ry(-1.835845545739718) q[2];
cx q[1],q[2];
ry(-0.017092001525140788) q[1];
ry(-1.6311748004539226) q[2];
cx q[1],q[2];
ry(-2.8154206875309487) q[2];
ry(2.496824710662436) q[3];
cx q[2],q[3];
ry(-3.098700795180303) q[2];
ry(-0.4745849593064122) q[3];
cx q[2],q[3];
ry(0.0024937781259462) q[3];
ry(1.7671606818431727) q[4];
cx q[3],q[4];
ry(1.350555432282258) q[3];
ry(-0.4833023709029592) q[4];
cx q[3],q[4];
ry(1.6937282827439164) q[4];
ry(-2.9550075768641317) q[5];
cx q[4],q[5];
ry(1.501094340281082) q[4];
ry(-0.014903521682824356) q[5];
cx q[4],q[5];
ry(0.03902466664191807) q[5];
ry(1.595752009537591) q[6];
cx q[5],q[6];
ry(-1.534663315636969) q[5];
ry(2.280478767975416) q[6];
cx q[5],q[6];
ry(-0.4409183210167127) q[6];
ry(-1.9390271856974701) q[7];
cx q[6],q[7];
ry(-3.0627926365310336) q[6];
ry(-3.121672413662539) q[7];
cx q[6],q[7];
ry(1.3581899507870538) q[7];
ry(-0.6884714022140024) q[8];
cx q[7],q[8];
ry(-1.2186785777321898) q[7];
ry(0.144917451663999) q[8];
cx q[7],q[8];
ry(-0.5174453760346198) q[8];
ry(-2.6016732358874854) q[9];
cx q[8],q[9];
ry(1.4185229488880946) q[8];
ry(-0.6073915357886613) q[9];
cx q[8],q[9];
ry(-1.2549048101923725) q[9];
ry(-2.6902544698335156) q[10];
cx q[9],q[10];
ry(1.5278928674138976) q[9];
ry(-1.6483695698000624) q[10];
cx q[9],q[10];
ry(-1.116772092538275) q[10];
ry(-1.0490415224718448) q[11];
cx q[10],q[11];
ry(-1.874682477995112) q[10];
ry(-3.1109312015322326) q[11];
cx q[10],q[11];
ry(2.5094610358647387) q[0];
ry(-2.857094349509945) q[1];
cx q[0],q[1];
ry(1.783323012281602) q[0];
ry(2.7726998144363137) q[1];
cx q[0],q[1];
ry(3.140197909596243) q[1];
ry(-1.4978541199635353) q[2];
cx q[1],q[2];
ry(-0.22424341767733313) q[1];
ry(-0.20053062626079718) q[2];
cx q[1],q[2];
ry(1.3842974965334447) q[2];
ry(2.32731219196747) q[3];
cx q[2],q[3];
ry(2.293801164258618) q[2];
ry(0.5705575316209428) q[3];
cx q[2],q[3];
ry(-0.5147431286387434) q[3];
ry(-1.6700894098120012) q[4];
cx q[3],q[4];
ry(3.1184370484941755) q[3];
ry(-0.14474057009026997) q[4];
cx q[3],q[4];
ry(-3.0489721090438366) q[4];
ry(0.9432059664196201) q[5];
cx q[4],q[5];
ry(-1.8606725408891736) q[4];
ry(-3.103677952502043) q[5];
cx q[4],q[5];
ry(-2.976210472811683) q[5];
ry(0.8949375244285668) q[6];
cx q[5],q[6];
ry(-0.136945768860222) q[5];
ry(1.909338388836215) q[6];
cx q[5],q[6];
ry(-0.6606657758720463) q[6];
ry(1.1563233350294313) q[7];
cx q[6],q[7];
ry(0.04430144787687402) q[6];
ry(0.12182140983420232) q[7];
cx q[6],q[7];
ry(-2.536930277280563) q[7];
ry(-2.72229286555172) q[8];
cx q[7],q[8];
ry(0.6480326567624983) q[7];
ry(0.3236038133268212) q[8];
cx q[7],q[8];
ry(3.123120813847338) q[8];
ry(1.5633688744749212) q[9];
cx q[8],q[9];
ry(1.4475555121675834) q[8];
ry(-1.5572052086481685) q[9];
cx q[8],q[9];
ry(-1.403777892960684) q[9];
ry(-2.896704673726493) q[10];
cx q[9],q[10];
ry(-1.0977356589259908) q[9];
ry(-2.3490556207040356) q[10];
cx q[9],q[10];
ry(-1.0218074958200143) q[10];
ry(0.12443512007802213) q[11];
cx q[10],q[11];
ry(1.3598341229009574) q[10];
ry(0.9311475427280858) q[11];
cx q[10],q[11];
ry(2.1858234263265173) q[0];
ry(2.968681849047677) q[1];
cx q[0],q[1];
ry(0.00015400413331612128) q[0];
ry(-2.184322492214808) q[1];
cx q[0],q[1];
ry(-3.0792729845511935) q[1];
ry(0.033399591581110485) q[2];
cx q[1],q[2];
ry(3.138832838594998) q[1];
ry(0.4612571075558681) q[2];
cx q[1],q[2];
ry(1.8656838093220705) q[2];
ry(0.14667314723348376) q[3];
cx q[2],q[3];
ry(0.16802101938218428) q[2];
ry(2.63179370518839) q[3];
cx q[2],q[3];
ry(2.0734341591402856) q[3];
ry(1.5754392882732358) q[4];
cx q[3],q[4];
ry(3.1000459746154494) q[3];
ry(-3.0191865448763644) q[4];
cx q[3],q[4];
ry(-1.6500317365422612) q[4];
ry(1.4728197349146566) q[5];
cx q[4],q[5];
ry(2.3837720936140827) q[4];
ry(1.6924825464719064) q[5];
cx q[4],q[5];
ry(0.7320503077027533) q[5];
ry(0.8934350486325924) q[6];
cx q[5],q[6];
ry(1.4834656149938519) q[5];
ry(-2.18563700608712) q[6];
cx q[5],q[6];
ry(2.4997169270029347) q[6];
ry(-2.8619326593159284) q[7];
cx q[6],q[7];
ry(3.1232901339027115) q[6];
ry(-0.21395049188240814) q[7];
cx q[6],q[7];
ry(2.531352578414994) q[7];
ry(-2.806297504408587) q[8];
cx q[7],q[8];
ry(1.3143971450674494) q[7];
ry(-3.0924050826403904) q[8];
cx q[7],q[8];
ry(1.6189041472913737) q[8];
ry(-0.9096994313519584) q[9];
cx q[8],q[9];
ry(-3.1143756825910263) q[8];
ry(-0.10777758800321283) q[9];
cx q[8],q[9];
ry(-0.9057667718133998) q[9];
ry(-0.34877575834212005) q[10];
cx q[9],q[10];
ry(0.006482207242067162) q[9];
ry(0.598293035895729) q[10];
cx q[9],q[10];
ry(0.4622284844844877) q[10];
ry(1.5885932825946545) q[11];
cx q[10],q[11];
ry(-1.3420685962302903) q[10];
ry(2.382238873191099) q[11];
cx q[10],q[11];
ry(-1.872105019881615) q[0];
ry(2.591646041367924) q[1];
cx q[0],q[1];
ry(-1.8026218487701744) q[0];
ry(1.5786609017041329) q[1];
cx q[0],q[1];
ry(-0.9671827139647521) q[1];
ry(1.6409233815280617) q[2];
cx q[1],q[2];
ry(3.131842287391231) q[1];
ry(3.04903100311161) q[2];
cx q[1],q[2];
ry(1.032883124263865) q[2];
ry(1.781821334837785) q[3];
cx q[2],q[3];
ry(-0.549352336663454) q[2];
ry(2.21332108678215) q[3];
cx q[2],q[3];
ry(2.605713699042368) q[3];
ry(-1.5331306286324384) q[4];
cx q[3],q[4];
ry(0.04034135695556652) q[3];
ry(3.140896387844174) q[4];
cx q[3],q[4];
ry(1.5354792018213184) q[4];
ry(-1.580385249396718) q[5];
cx q[4],q[5];
ry(-1.6232854535246106) q[4];
ry(-1.792335694155284) q[5];
cx q[4],q[5];
ry(-2.9657443573732984) q[5];
ry(-1.9582033510943617) q[6];
cx q[5],q[6];
ry(-0.22169426643523682) q[5];
ry(1.5739744004081024) q[6];
cx q[5],q[6];
ry(-3.0071288868995967) q[6];
ry(-0.7329369148366788) q[7];
cx q[6],q[7];
ry(0.003383187329762727) q[6];
ry(-3.140051841160054) q[7];
cx q[6],q[7];
ry(3.102136015696069) q[7];
ry(-0.5071153793603276) q[8];
cx q[7],q[8];
ry(-1.3668285849703175) q[7];
ry(0.5538369875357008) q[8];
cx q[7],q[8];
ry(-0.02798901410507515) q[8];
ry(-1.5152873023290585) q[9];
cx q[8],q[9];
ry(1.4722521737724827) q[8];
ry(-3.094722575622172) q[9];
cx q[8],q[9];
ry(1.553338745033723) q[9];
ry(-0.8222295946807865) q[10];
cx q[9],q[10];
ry(1.3924630143817485) q[9];
ry(1.914344436358478) q[10];
cx q[9],q[10];
ry(-0.7148205938889909) q[10];
ry(1.2424004191600784) q[11];
cx q[10],q[11];
ry(-3.0917501813111303) q[10];
ry(-1.3664085520345584) q[11];
cx q[10],q[11];
ry(-0.23398805787288318) q[0];
ry(2.1220166202551454) q[1];
cx q[0],q[1];
ry(-2.1456828688041156) q[0];
ry(1.60478061025377) q[1];
cx q[0],q[1];
ry(-0.19741168487525035) q[1];
ry(-0.6406560450565095) q[2];
cx q[1],q[2];
ry(-1.0335443280551637) q[1];
ry(0.2402314484357587) q[2];
cx q[1],q[2];
ry(-1.8327986080339764) q[2];
ry(0.5739492859221711) q[3];
cx q[2],q[3];
ry(0.0065113116020372175) q[2];
ry(0.1068369063655128) q[3];
cx q[2],q[3];
ry(3.0906253048745937) q[3];
ry(-3.0787753083781535) q[4];
cx q[3],q[4];
ry(-0.025125687747216678) q[3];
ry(3.139120506886445) q[4];
cx q[3],q[4];
ry(-1.908074611533639) q[4];
ry(-0.11505913834617854) q[5];
cx q[4],q[5];
ry(-0.01446762068490326) q[4];
ry(-1.5158483879917641) q[5];
cx q[4],q[5];
ry(1.6067121308895986) q[5];
ry(-1.8225342496490275) q[6];
cx q[5],q[6];
ry(-1.6741237977863654) q[5];
ry(-0.011949031461110714) q[6];
cx q[5],q[6];
ry(-0.8743909839231857) q[6];
ry(-2.137604428541013) q[7];
cx q[6],q[7];
ry(0.002175237754190712) q[6];
ry(0.8700560636260475) q[7];
cx q[6],q[7];
ry(1.2734907444771895) q[7];
ry(1.2387847834503036) q[8];
cx q[7],q[8];
ry(1.5195146428647455) q[7];
ry(3.1408221426198155) q[8];
cx q[7],q[8];
ry(1.5996226670560985) q[8];
ry(1.5415082694335682) q[9];
cx q[8],q[9];
ry(-1.7253834813302174) q[8];
ry(-1.9894237499458294) q[9];
cx q[8],q[9];
ry(-1.2555563594819015) q[9];
ry(0.28801584511846395) q[10];
cx q[9],q[10];
ry(-0.5741792610991681) q[9];
ry(3.1139378174910397) q[10];
cx q[9],q[10];
ry(-1.4760507151430766) q[10];
ry(-2.2958987525669494) q[11];
cx q[10],q[11];
ry(-1.7737681088106976) q[10];
ry(0.18349927004523892) q[11];
cx q[10],q[11];
ry(3.062184546036729) q[0];
ry(1.3394302258728628) q[1];
cx q[0],q[1];
ry(0.7117567800629985) q[0];
ry(-0.011724549022117316) q[1];
cx q[0],q[1];
ry(-2.3891855638148987) q[1];
ry(-2.4901263011424515) q[2];
cx q[1],q[2];
ry(-2.213138125701996) q[1];
ry(1.974787492777292) q[2];
cx q[1],q[2];
ry(2.656486946846079) q[2];
ry(-2.3583715693368035) q[3];
cx q[2],q[3];
ry(3.092303246506493) q[2];
ry(-3.0192440610581954) q[3];
cx q[2],q[3];
ry(-2.5940287209920605) q[3];
ry(-0.6507484536059955) q[4];
cx q[3],q[4];
ry(0.04104328534790283) q[3];
ry(-0.006937388340273464) q[4];
cx q[3],q[4];
ry(-2.071843250099189) q[4];
ry(-1.4820319143743141) q[5];
cx q[4],q[5];
ry(-1.7307902775525443) q[4];
ry(1.725692610548168) q[5];
cx q[4],q[5];
ry(-2.146684786551173) q[5];
ry(-1.4378387588614738) q[6];
cx q[5],q[6];
ry(-0.017358960567924523) q[5];
ry(3.13758588229012) q[6];
cx q[5],q[6];
ry(-0.2759010010642308) q[6];
ry(0.6410550191890803) q[7];
cx q[6],q[7];
ry(1.55904192607652) q[6];
ry(2.045829633084935) q[7];
cx q[6],q[7];
ry(-1.8195789192943355) q[7];
ry(1.6218263731726443) q[8];
cx q[7],q[8];
ry(2.8909278728830246) q[7];
ry(3.1309342708347594) q[8];
cx q[7],q[8];
ry(-1.8130789858046192) q[8];
ry(-1.2720878793010684) q[9];
cx q[8],q[9];
ry(0.32382241246634047) q[8];
ry(0.001222301717874904) q[9];
cx q[8],q[9];
ry(3.0200758255184303) q[9];
ry(-1.4398025828736536) q[10];
cx q[9],q[10];
ry(-2.7392810297881285) q[9];
ry(-0.3743887416274019) q[10];
cx q[9],q[10];
ry(2.638030804695666) q[10];
ry(-0.012493253546411563) q[11];
cx q[10],q[11];
ry(2.1046260777851016) q[10];
ry(-0.028032178386572022) q[11];
cx q[10],q[11];
ry(-0.8213888671489178) q[0];
ry(2.325663255133872) q[1];
cx q[0],q[1];
ry(-0.5201685345470075) q[0];
ry(-1.498000448283059) q[1];
cx q[0],q[1];
ry(-0.8030854402158127) q[1];
ry(-1.7053209448242201) q[2];
cx q[1],q[2];
ry(-2.749726186535897) q[1];
ry(3.1250617057742955) q[2];
cx q[1],q[2];
ry(-0.47491559152785534) q[2];
ry(0.313979551637416) q[3];
cx q[2],q[3];
ry(1.6088900424960686) q[2];
ry(1.604319854606022) q[3];
cx q[2],q[3];
ry(-2.8106096869861164) q[3];
ry(2.9152095163171943) q[4];
cx q[3],q[4];
ry(1.5794883527591588) q[3];
ry(3.141477342984386) q[4];
cx q[3],q[4];
ry(-1.5515198045949292) q[4];
ry(0.9922313349472232) q[5];
cx q[4],q[5];
ry(1.562096959854344) q[4];
ry(2.6438596500043072) q[5];
cx q[4],q[5];
ry(-1.574433843904013) q[5];
ry(2.448174668053242) q[6];
cx q[5],q[6];
ry(-1.5750969952802574) q[5];
ry(0.15931222651697308) q[6];
cx q[5],q[6];
ry(1.5728459731918436) q[6];
ry(1.312114269551547) q[7];
cx q[6],q[7];
ry(-3.0444307955204626) q[6];
ry(2.949038670283473) q[7];
cx q[6],q[7];
ry(1.561635504045162) q[7];
ry(1.8565806093662016) q[8];
cx q[7],q[8];
ry(1.455745047106019) q[7];
ry(1.6257774727726853) q[8];
cx q[7],q[8];
ry(0.20932590626550063) q[8];
ry(-2.7968056718606373) q[9];
cx q[8],q[9];
ry(-3.1167774655570324) q[8];
ry(-0.006270288889153264) q[9];
cx q[8],q[9];
ry(-2.7324261468639146) q[9];
ry(0.6555111600117021) q[10];
cx q[9],q[10];
ry(0.5530433380085515) q[9];
ry(2.7793414738061735) q[10];
cx q[9],q[10];
ry(0.669825758924172) q[10];
ry(1.6562561948925492) q[11];
cx q[10],q[11];
ry(-0.3203309948754267) q[10];
ry(2.029146063745712) q[11];
cx q[10],q[11];
ry(-0.7783212374420724) q[0];
ry(0.3014334658933784) q[1];
cx q[0],q[1];
ry(-2.118643044330958) q[0];
ry(-0.7355620821231383) q[1];
cx q[0],q[1];
ry(-2.132321352405465) q[1];
ry(-1.3764091811659878) q[2];
cx q[1],q[2];
ry(-0.02476435693485257) q[1];
ry(0.5363566550653555) q[2];
cx q[1],q[2];
ry(1.312365106080918) q[2];
ry(0.29165704946031923) q[3];
cx q[2],q[3];
ry(1.4654558788689025) q[2];
ry(-1.272154518962283) q[3];
cx q[2],q[3];
ry(-1.5703000584598954) q[3];
ry(-1.5558889411926327) q[4];
cx q[3],q[4];
ry(-2.4508886423948635) q[3];
ry(-0.644352751757545) q[4];
cx q[3],q[4];
ry(1.5726002778322457) q[4];
ry(-2.9811908640173805) q[5];
cx q[4],q[5];
ry(-3.139364286519039) q[4];
ry(-1.5339303219697304) q[5];
cx q[4],q[5];
ry(0.18844867628318787) q[5];
ry(-0.41535921833930317) q[6];
cx q[5],q[6];
ry(0.004383049738026834) q[5];
ry(-1.8658558808721075) q[6];
cx q[5],q[6];
ry(-1.9528902138165416) q[6];
ry(-1.5673956127393363) q[7];
cx q[6],q[7];
ry(-1.559736285439891) q[6];
ry(0.006208462871120581) q[7];
cx q[6],q[7];
ry(-1.5742488856230903) q[7];
ry(2.922108277739912) q[8];
cx q[7],q[8];
ry(1.6486151355391054) q[7];
ry(0.0558298481683385) q[8];
cx q[7],q[8];
ry(-1.5181901758090341) q[8];
ry(-3.013257868652832) q[9];
cx q[8],q[9];
ry(1.8948598347738743) q[8];
ry(-1.03187527025373) q[9];
cx q[8],q[9];
ry(0.6943465948229395) q[9];
ry(2.2988271998813308) q[10];
cx q[9],q[10];
ry(0.0004388918593187796) q[9];
ry(-0.0010399218095500373) q[10];
cx q[9],q[10];
ry(-0.5323894361046843) q[10];
ry(-0.49457647119387693) q[11];
cx q[10],q[11];
ry(2.6076594104137856) q[10];
ry(2.21646568349416) q[11];
cx q[10],q[11];
ry(2.5062267766524475) q[0];
ry(1.9433810518097625) q[1];
cx q[0],q[1];
ry(3.0643418180252726) q[0];
ry(-1.4668126031525457) q[1];
cx q[0],q[1];
ry(0.8938545779962528) q[1];
ry(1.457740782826777) q[2];
cx q[1],q[2];
ry(1.476490760855642) q[1];
ry(-2.3377460583376375) q[2];
cx q[1],q[2];
ry(0.727231918144132) q[2];
ry(1.5743068053798257) q[3];
cx q[2],q[3];
ry(0.23643749992459992) q[2];
ry(3.0991316081720273) q[3];
cx q[2],q[3];
ry(1.8573341438192328) q[3];
ry(1.5678432358081151) q[4];
cx q[3],q[4];
ry(-0.32534110658541415) q[3];
ry(-0.017032510958535627) q[4];
cx q[3],q[4];
ry(1.3878792151237338) q[4];
ry(-1.4337387754771909) q[5];
cx q[4],q[5];
ry(-0.9718602217497273) q[4];
ry(3.0532934864463046) q[5];
cx q[4],q[5];
ry(-1.560522900224186) q[5];
ry(-2.3168071539975066) q[6];
cx q[5],q[6];
ry(3.1361641936035816) q[5];
ry(-2.859120651717516) q[6];
cx q[5],q[6];
ry(0.4272261269566763) q[6];
ry(2.7644854378302557) q[7];
cx q[6],q[7];
ry(-0.0014039967918852978) q[6];
ry(1.7357881436676017) q[7];
cx q[6],q[7];
ry(2.0376777035419584) q[7];
ry(1.561472765395377) q[8];
cx q[7],q[8];
ry(-0.11801555547243467) q[7];
ry(-9.30892238537595e-05) q[8];
cx q[7],q[8];
ry(-1.562576400383857) q[8];
ry(-0.943886949297145) q[9];
cx q[8],q[9];
ry(2.7933392039984297) q[8];
ry(-2.1014517951052234) q[9];
cx q[8],q[9];
ry(-1.8429561481237153) q[9];
ry(-2.2923790165602576) q[10];
cx q[9],q[10];
ry(-2.140019143119005) q[9];
ry(-0.7512715712959714) q[10];
cx q[9],q[10];
ry(-2.5203825267139455) q[10];
ry(-1.4645654925368623) q[11];
cx q[10],q[11];
ry(1.8354697284631136) q[10];
ry(-3.1415187163796907) q[11];
cx q[10],q[11];
ry(-2.4116171352119093) q[0];
ry(1.7485505464451367) q[1];
cx q[0],q[1];
ry(-2.419272867879145) q[0];
ry(-3.037731450067783) q[1];
cx q[0],q[1];
ry(-1.5825546108288175) q[1];
ry(0.1755079050455458) q[2];
cx q[1],q[2];
ry(-3.0578557077581636) q[1];
ry(-0.8757733791920606) q[2];
cx q[1],q[2];
ry(-2.122229305983251) q[2];
ry(-1.8613940464168364) q[3];
cx q[2],q[3];
ry(1.7363603405437047) q[2];
ry(1.5755243745194356) q[3];
cx q[2],q[3];
ry(1.8547954872838037) q[3];
ry(1.447447918415329) q[4];
cx q[3],q[4];
ry(-0.004499536581077619) q[3];
ry(-3.121763903111918) q[4];
cx q[3],q[4];
ry(-1.8777513049645753) q[4];
ry(-1.5622645157445423) q[5];
cx q[4],q[5];
ry(-2.185187222042715) q[4];
ry(-0.06862686846928374) q[5];
cx q[4],q[5];
ry(2.996914472602091) q[5];
ry(-3.1041217353643535) q[6];
cx q[5],q[6];
ry(3.0690798924356955) q[5];
ry(0.09191548141188566) q[6];
cx q[5],q[6];
ry(-1.558176423365315) q[6];
ry(3.098607279030733) q[7];
cx q[6],q[7];
ry(0.0016490158721902404) q[6];
ry(1.1691606576169038) q[7];
cx q[6],q[7];
ry(1.0207351554077846) q[7];
ry(-1.530355124832715) q[8];
cx q[7],q[8];
ry(0.32308981306540746) q[7];
ry(-2.2858431394188905) q[8];
cx q[7],q[8];
ry(-1.1210937407111476) q[8];
ry(-2.327276993958524) q[9];
cx q[8],q[9];
ry(0.0031479632382904977) q[8];
ry(-3.1381355789859424) q[9];
cx q[8],q[9];
ry(-2.55730145018839) q[9];
ry(0.6199477643479713) q[10];
cx q[9],q[10];
ry(1.9679341867597717) q[9];
ry(3.115710743529516) q[10];
cx q[9],q[10];
ry(1.60226468292196) q[10];
ry(-2.1822126536823734) q[11];
cx q[10],q[11];
ry(-0.23734600400183847) q[10];
ry(-0.2574415602185116) q[11];
cx q[10],q[11];
ry(-1.5636124267292066) q[0];
ry(-3.022470614686117) q[1];
cx q[0],q[1];
ry(-1.7785024746578646) q[0];
ry(-0.0094376271189196) q[1];
cx q[0],q[1];
ry(-1.7490143824673323) q[1];
ry(1.7203945157560927) q[2];
cx q[1],q[2];
ry(1.4056206155913333) q[1];
ry(0.2741306230430076) q[2];
cx q[1],q[2];
ry(-1.0957004291624006) q[2];
ry(-1.336940623355816) q[3];
cx q[2],q[3];
ry(0.0012644566732575267) q[2];
ry(-3.1330923430697215) q[3];
cx q[2],q[3];
ry(0.07018217455855424) q[3];
ry(-1.0124862963433525) q[4];
cx q[3],q[4];
ry(0.5477524719028626) q[3];
ry(-1.546726677609018) q[4];
cx q[3],q[4];
ry(-0.17499678042636457) q[4];
ry(-2.7053452806156213) q[5];
cx q[4],q[5];
ry(-0.002623075745139758) q[4];
ry(-3.098651713831555) q[5];
cx q[4],q[5];
ry(-2.032798461035008) q[5];
ry(-1.413779055379632) q[6];
cx q[5],q[6];
ry(-1.6497980476339764) q[5];
ry(-1.8710423075181595) q[6];
cx q[5],q[6];
ry(0.23887290066572042) q[6];
ry(-0.34553552069229904) q[7];
cx q[6],q[7];
ry(-0.18948262336369395) q[6];
ry(-0.0011663785301969654) q[7];
cx q[6],q[7];
ry(1.2420065181300153) q[7];
ry(2.0049193724535415) q[8];
cx q[7],q[8];
ry(2.259713008686208) q[7];
ry(-3.04917036731688) q[8];
cx q[7],q[8];
ry(2.509978658169079) q[8];
ry(-1.1500023709867042) q[9];
cx q[8],q[9];
ry(-1.7263510277548209) q[8];
ry(3.1384494731488526) q[9];
cx q[8],q[9];
ry(-0.19518296452855116) q[9];
ry(2.876215824341081) q[10];
cx q[9],q[10];
ry(-0.0006484409885745279) q[9];
ry(3.1369945300429056) q[10];
cx q[9],q[10];
ry(-1.1234448113948774) q[10];
ry(-2.7612739298369657) q[11];
cx q[10],q[11];
ry(1.596031483254237) q[10];
ry(-0.6112301720555999) q[11];
cx q[10],q[11];
ry(-0.9108571819384697) q[0];
ry(0.9337669563254609) q[1];
cx q[0],q[1];
ry(1.9131975714900298) q[0];
ry(3.1285584337963668) q[1];
cx q[0],q[1];
ry(0.44773889006146916) q[1];
ry(2.608472167377247) q[2];
cx q[1],q[2];
ry(-1.3121324386035722) q[1];
ry(-2.3352439499945765) q[2];
cx q[1],q[2];
ry(0.5123505416869288) q[2];
ry(1.703159679959985) q[3];
cx q[2],q[3];
ry(3.1409945532303247) q[2];
ry(0.03630058531529613) q[3];
cx q[2],q[3];
ry(0.16551057203225295) q[3];
ry(-1.1641235244213863) q[4];
cx q[3],q[4];
ry(-0.18247824933670653) q[3];
ry(1.5856583167307832) q[4];
cx q[3],q[4];
ry(2.8576221859982254) q[4];
ry(-3.102999215362804) q[5];
cx q[4],q[5];
ry(3.104967695379667) q[4];
ry(3.0776134587385005) q[5];
cx q[4],q[5];
ry(-3.006470891214399) q[5];
ry(-0.13380297420266785) q[6];
cx q[5],q[6];
ry(1.5616049650381187) q[5];
ry(-2.6526516394066495) q[6];
cx q[5],q[6];
ry(1.9246958217694816) q[6];
ry(0.32682562439089136) q[7];
cx q[6],q[7];
ry(0.24673284516857577) q[6];
ry(-6.443376341103146e-05) q[7];
cx q[6],q[7];
ry(0.16817049591636124) q[7];
ry(0.7171533082114552) q[8];
cx q[7],q[8];
ry(0.00310148932142198) q[7];
ry(-0.40078243582769585) q[8];
cx q[7],q[8];
ry(-0.15576415276098976) q[8];
ry(1.5584040458983557) q[9];
cx q[8],q[9];
ry(-0.9348518520794014) q[8];
ry(3.135585539184867) q[9];
cx q[8],q[9];
ry(-3.1140328002032978) q[9];
ry(3.100100413110883) q[10];
cx q[9],q[10];
ry(3.1404958532231064) q[9];
ry(2.266674503899574) q[10];
cx q[9],q[10];
ry(0.24050825389311653) q[10];
ry(-0.5374075504919791) q[11];
cx q[10],q[11];
ry(-2.8306458892576165) q[10];
ry(-1.9936101331718126) q[11];
cx q[10],q[11];
ry(1.0228873698461483) q[0];
ry(1.3460126199513773) q[1];
cx q[0],q[1];
ry(-2.73998864473715) q[0];
ry(-3.0968009047778735) q[1];
cx q[0],q[1];
ry(2.405441894547138) q[1];
ry(2.2822860248687027) q[2];
cx q[1],q[2];
ry(1.3414740891227843) q[1];
ry(-1.7409990163206617) q[2];
cx q[1],q[2];
ry(-1.057187154414228) q[2];
ry(2.0178695839720175) q[3];
cx q[2],q[3];
ry(-0.0008681341905560913) q[2];
ry(-2.5313993918973) q[3];
cx q[2],q[3];
ry(-0.42975515977891826) q[3];
ry(1.5442005647140622) q[4];
cx q[3],q[4];
ry(2.8524895567634694) q[3];
ry(-1.600466478038279) q[4];
cx q[3],q[4];
ry(-1.6155224151791756) q[4];
ry(-1.5128150181315396) q[5];
cx q[4],q[5];
ry(1.547065062386613) q[4];
ry(-1.5673784644490114) q[5];
cx q[4],q[5];
ry(1.5714431287536927) q[5];
ry(-1.235288313848205) q[6];
cx q[5],q[6];
ry(-3.136704115289328) q[5];
ry(1.0014092927053113) q[6];
cx q[5],q[6];
ry(-2.4593876530925187) q[6];
ry(-0.7654124574223848) q[7];
cx q[6],q[7];
ry(-1.507624509705364) q[6];
ry(-0.0036333394045893644) q[7];
cx q[6],q[7];
ry(1.6726110859045182) q[7];
ry(1.1921712343908584) q[8];
cx q[7],q[8];
ry(-1.0565406183604664) q[7];
ry(1.8198651335592224) q[8];
cx q[7],q[8];
ry(-1.5826447407096378) q[8];
ry(-1.4818378387836235) q[9];
cx q[8],q[9];
ry(1.591658317339291) q[8];
ry(1.4829316468545901) q[9];
cx q[8],q[9];
ry(2.9865054203388794) q[9];
ry(-0.346511501655387) q[10];
cx q[9],q[10];
ry(-9.750986923506844e-05) q[9];
ry(0.04842698362367167) q[10];
cx q[9],q[10];
ry(2.2710203821861197) q[10];
ry(0.6947710497390518) q[11];
cx q[10],q[11];
ry(-1.9254395405457774) q[10];
ry(-1.5625867864032186) q[11];
cx q[10],q[11];
ry(1.5465608133389295) q[0];
ry(-2.541142433663486) q[1];
cx q[0],q[1];
ry(-0.14485185540296575) q[0];
ry(-1.5630324345826334) q[1];
cx q[0],q[1];
ry(-3.0829127072513365) q[1];
ry(-1.5706243800902175) q[2];
cx q[1],q[2];
ry(2.5376814985883427) q[1];
ry(1.462201277837507) q[2];
cx q[1],q[2];
ry(1.5672871173895784) q[2];
ry(-1.5742281614858182) q[3];
cx q[2],q[3];
ry(1.4096487976837535) q[2];
ry(1.1867218735881393) q[3];
cx q[2],q[3];
ry(2.484653555585797) q[3];
ry(1.5704717081001585) q[4];
cx q[3],q[4];
ry(-0.6074442065980489) q[3];
ry(3.1413532537079925) q[4];
cx q[3],q[4];
ry(-2.5732200972374364) q[4];
ry(1.57081585492294) q[5];
cx q[4],q[5];
ry(1.5870074866979722) q[4];
ry(-3.1415098251918527) q[5];
cx q[4],q[5];
ry(1.5711906181604114) q[5];
ry(1.6527670291173118) q[6];
cx q[5],q[6];
ry(-3.140278723386819) q[5];
ry(-2.571802623725244) q[6];
cx q[5],q[6];
ry(-1.498547532354335) q[6];
ry(-1.6677388984203843) q[7];
cx q[6],q[7];
ry(1.7793142144525147) q[6];
ry(-1.5609754390354902) q[7];
cx q[6],q[7];
ry(1.5975734541413555) q[7];
ry(-1.9342157017644805) q[8];
cx q[7],q[8];
ry(3.1395876488372507) q[7];
ry(3.1412775677715357) q[8];
cx q[7],q[8];
ry(1.2068004063547642) q[8];
ry(0.11086886653246547) q[9];
cx q[8],q[9];
ry(-2.079251571780464) q[8];
ry(-1.6741297590916364) q[9];
cx q[8],q[9];
ry(1.4818750167837944) q[9];
ry(-2.8636906077683215) q[10];
cx q[9],q[10];
ry(0.019113866330241258) q[9];
ry(1.558566183040255) q[10];
cx q[9],q[10];
ry(-1.2962917190460104) q[10];
ry(-1.4677004190299072) q[11];
cx q[10],q[11];
ry(1.4104834698210844) q[10];
ry(2.429008885146148) q[11];
cx q[10],q[11];
ry(-1.5865756847452595) q[0];
ry(1.5731446912358908) q[1];
ry(-1.5695212627118345) q[2];
ry(-0.6582858354092397) q[3];
ry(0.5679346666950207) q[4];
ry(-1.5704111961417526) q[5];
ry(1.5707050952474544) q[6];
ry(1.5981623456933778) q[7];
ry(-1.5597373279960944) q[8];
ry(1.5717989633499572) q[9];
ry(2.282021985456724) q[10];
ry(1.1564514377413566) q[11];