OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.213503000931931) q[0];
ry(-0.9123363580887389) q[1];
cx q[0],q[1];
ry(-2.4634659600676545) q[0];
ry(3.066878149270362) q[1];
cx q[0],q[1];
ry(0.6730872410863623) q[2];
ry(-2.349632572204547) q[3];
cx q[2],q[3];
ry(1.500553992430345) q[2];
ry(-1.6781111057269689) q[3];
cx q[2],q[3];
ry(2.678224358014023) q[4];
ry(-2.2662294985029092) q[5];
cx q[4],q[5];
ry(-2.4947534035180703) q[4];
ry(-2.929190683987696) q[5];
cx q[4],q[5];
ry(-2.882665737690142) q[6];
ry(2.367315789411307) q[7];
cx q[6],q[7];
ry(-0.9475417317122803) q[6];
ry(-1.9944631412260825) q[7];
cx q[6],q[7];
ry(2.717579405638244) q[8];
ry(2.946498879507217) q[9];
cx q[8],q[9];
ry(-1.4266868223306364) q[8];
ry(-1.816883958247367) q[9];
cx q[8],q[9];
ry(-1.9712484199212348) q[10];
ry(0.049789738339245077) q[11];
cx q[10],q[11];
ry(1.1406979158237298) q[10];
ry(-0.07724695819563898) q[11];
cx q[10],q[11];
ry(-2.580680536343961) q[12];
ry(-1.2515477118460945) q[13];
cx q[12],q[13];
ry(2.984692161065122) q[12];
ry(0.10985095563122513) q[13];
cx q[12],q[13];
ry(2.598593205527634) q[14];
ry(0.2181565277498989) q[15];
cx q[14],q[15];
ry(-0.8307331313431581) q[14];
ry(2.1658635787471883) q[15];
cx q[14],q[15];
ry(2.0377642769342867) q[16];
ry(-2.408321239691004) q[17];
cx q[16],q[17];
ry(2.948889459747808) q[16];
ry(0.16366254745365447) q[17];
cx q[16],q[17];
ry(-2.7714361311162885) q[18];
ry(-2.7129423564315447) q[19];
cx q[18],q[19];
ry(1.0361242374082291) q[18];
ry(-2.8502516836904075) q[19];
cx q[18],q[19];
ry(0.9663739501511676) q[0];
ry(0.5773374514769134) q[2];
cx q[0],q[2];
ry(-1.5760675355153237) q[0];
ry(-2.1113335164455007) q[2];
cx q[0],q[2];
ry(-2.501531450041136) q[2];
ry(0.14269489942960256) q[4];
cx q[2],q[4];
ry(-0.9100024013562286) q[2];
ry(2.5489586733779257) q[4];
cx q[2],q[4];
ry(2.1216210227856065) q[4];
ry(0.7542781526682889) q[6];
cx q[4],q[6];
ry(-0.06064683933308914) q[4];
ry(2.9879941687589864) q[6];
cx q[4],q[6];
ry(-2.563089845871808) q[6];
ry(-3.098379345957189) q[8];
cx q[6],q[8];
ry(-1.971502162070335) q[6];
ry(3.1182126071368237) q[8];
cx q[6],q[8];
ry(1.9254581894190124) q[8];
ry(-1.5260500611294858) q[10];
cx q[8],q[10];
ry(2.438358244208238) q[8];
ry(-2.701208144210979) q[10];
cx q[8],q[10];
ry(-2.836670655628422) q[10];
ry(1.697537341416826) q[12];
cx q[10],q[12];
ry(-3.138974282424311) q[10];
ry(0.006087649318640082) q[12];
cx q[10],q[12];
ry(-0.9229157640767768) q[12];
ry(-1.738695141872422) q[14];
cx q[12],q[14];
ry(2.859997653465469) q[12];
ry(-0.3839735229881942) q[14];
cx q[12],q[14];
ry(0.0793189493360753) q[14];
ry(-0.7747230244748727) q[16];
cx q[14],q[16];
ry(-0.27160524370428973) q[14];
ry(-3.0892912666792216) q[16];
cx q[14],q[16];
ry(1.861235252737475) q[16];
ry(-2.7976522191680138) q[18];
cx q[16],q[18];
ry(-0.1708797800025577) q[16];
ry(-1.2750839615054876) q[18];
cx q[16],q[18];
ry(-2.091574677780571) q[1];
ry(1.7270812120937133) q[3];
cx q[1],q[3];
ry(1.9597135941070052) q[1];
ry(-1.4041773764308316) q[3];
cx q[1],q[3];
ry(0.7370864810413789) q[3];
ry(-0.12311252381441494) q[5];
cx q[3],q[5];
ry(-0.8716183108182819) q[3];
ry(-2.6176527551279873) q[5];
cx q[3],q[5];
ry(0.5050791724557311) q[5];
ry(-0.177608513438976) q[7];
cx q[5],q[7];
ry(-3.124426861294246) q[5];
ry(0.02902772753996752) q[7];
cx q[5],q[7];
ry(-2.3868281964738953) q[7];
ry(2.6539546160532876) q[9];
cx q[7],q[9];
ry(0.12409191717927914) q[7];
ry(0.030577098045713756) q[9];
cx q[7],q[9];
ry(1.8353357303397586) q[9];
ry(2.049804996167807) q[11];
cx q[9],q[11];
ry(-0.04577542240412766) q[9];
ry(0.22712990462345495) q[11];
cx q[9],q[11];
ry(-2.754413263062151) q[11];
ry(2.9874021920677003) q[13];
cx q[11],q[13];
ry(3.1255347727075753) q[11];
ry(0.014894319751132201) q[13];
cx q[11],q[13];
ry(-0.2573201505991316) q[13];
ry(0.7930998972763419) q[15];
cx q[13],q[15];
ry(1.3522010730499587) q[13];
ry(-2.666607971256781) q[15];
cx q[13],q[15];
ry(2.768633812408872) q[15];
ry(2.513748205573428) q[17];
cx q[15],q[17];
ry(-1.3918982434680727) q[15];
ry(-0.7897144181046681) q[17];
cx q[15],q[17];
ry(-1.418056783195843) q[17];
ry(-0.6812752986800783) q[19];
cx q[17],q[19];
ry(-0.8967336937634199) q[17];
ry(-1.0365652833029633) q[19];
cx q[17],q[19];
ry(0.7023563055503829) q[0];
ry(2.55384856581009) q[1];
cx q[0],q[1];
ry(2.8307044405619917) q[0];
ry(-0.4746055699550016) q[1];
cx q[0],q[1];
ry(1.6974194645724532) q[2];
ry(2.1343908252796293) q[3];
cx q[2],q[3];
ry(-0.8935319525885536) q[2];
ry(0.3229509031407805) q[3];
cx q[2],q[3];
ry(0.1410799031384915) q[4];
ry(2.6073481134616934) q[5];
cx q[4],q[5];
ry(1.3109797204787945) q[4];
ry(1.4674653153819897) q[5];
cx q[4],q[5];
ry(1.9681388237311541) q[6];
ry(-1.6942851442548819) q[7];
cx q[6],q[7];
ry(-2.0028660264187756) q[6];
ry(0.434643057555963) q[7];
cx q[6],q[7];
ry(1.3778695481573582) q[8];
ry(0.949869982766037) q[9];
cx q[8],q[9];
ry(-2.7321486491192974) q[8];
ry(3.074939480115483) q[9];
cx q[8],q[9];
ry(-0.7586747939404593) q[10];
ry(0.7546207389781809) q[11];
cx q[10],q[11];
ry(-2.6147587417957623) q[10];
ry(-2.966066595619791) q[11];
cx q[10],q[11];
ry(-2.9130930394991883) q[12];
ry(-1.5960281864396073) q[13];
cx q[12],q[13];
ry(1.0018770130735806) q[12];
ry(-2.787181697679355) q[13];
cx q[12],q[13];
ry(-2.0468132704900337) q[14];
ry(-3.037068537696892) q[15];
cx q[14],q[15];
ry(3.0792137545430234) q[14];
ry(2.238144449528736) q[15];
cx q[14],q[15];
ry(2.480789012606023) q[16];
ry(2.945545511277324) q[17];
cx q[16],q[17];
ry(2.752000397009147) q[16];
ry(2.5074668361663037) q[17];
cx q[16],q[17];
ry(0.6035627640268206) q[18];
ry(0.9758133025644167) q[19];
cx q[18],q[19];
ry(0.5291543196987059) q[18];
ry(2.3031647584857944) q[19];
cx q[18],q[19];
ry(1.4391857086632793) q[0];
ry(-2.059388190971328) q[2];
cx q[0],q[2];
ry(-0.15686563424668432) q[0];
ry(2.578668066969551) q[2];
cx q[0],q[2];
ry(0.9301437565142291) q[2];
ry(3.0517555181463827) q[4];
cx q[2],q[4];
ry(2.795343077965035) q[2];
ry(2.1206087052955223) q[4];
cx q[2],q[4];
ry(-1.0128967139820801) q[4];
ry(2.084931701379702) q[6];
cx q[4],q[6];
ry(3.050285802432665) q[4];
ry(-2.9691757100655845) q[6];
cx q[4],q[6];
ry(1.09129769344508) q[6];
ry(-3.0204361991165674) q[8];
cx q[6],q[8];
ry(3.134583771796507) q[6];
ry(0.026377401802131484) q[8];
cx q[6],q[8];
ry(-0.07448650554869207) q[8];
ry(-0.9613786183837663) q[10];
cx q[8],q[10];
ry(0.2966138311616265) q[8];
ry(3.0984891968108648) q[10];
cx q[8],q[10];
ry(-0.5865624620208054) q[10];
ry(0.9267403592423102) q[12];
cx q[10],q[12];
ry(-3.1019952146245067) q[10];
ry(0.0500564067245044) q[12];
cx q[10],q[12];
ry(-0.5935461846518971) q[12];
ry(1.8004071849699477) q[14];
cx q[12],q[14];
ry(0.3052213372239294) q[12];
ry(2.8855454379049696) q[14];
cx q[12],q[14];
ry(-0.0391632346096828) q[14];
ry(2.0034070935970734) q[16];
cx q[14],q[16];
ry(-0.041219375558386055) q[14];
ry(2.6619683893679627) q[16];
cx q[14],q[16];
ry(-1.1454872432289689) q[16];
ry(1.7053140862253786) q[18];
cx q[16],q[18];
ry(0.2321831505270815) q[16];
ry(-0.3015002266586539) q[18];
cx q[16],q[18];
ry(-2.830581239653403) q[1];
ry(1.7069819846508998) q[3];
cx q[1],q[3];
ry(-1.125422072969131) q[1];
ry(-1.2943766226557925) q[3];
cx q[1],q[3];
ry(-0.8569327598380515) q[3];
ry(-2.388405360346657) q[5];
cx q[3],q[5];
ry(3.025335369239277) q[3];
ry(-0.13403284144667926) q[5];
cx q[3],q[5];
ry(0.4071639500983384) q[5];
ry(2.0907177291855303) q[7];
cx q[5],q[7];
ry(0.144473747467092) q[5];
ry(3.122341690582411) q[7];
cx q[5],q[7];
ry(1.4148278674319414) q[7];
ry(-0.6610704414835991) q[9];
cx q[7],q[9];
ry(2.6674193840534244) q[7];
ry(-0.0036725723639383645) q[9];
cx q[7],q[9];
ry(-1.8842345649269503) q[9];
ry(-0.4986786194054443) q[11];
cx q[9],q[11];
ry(-3.0685181831157484) q[9];
ry(-0.03042300343428739) q[11];
cx q[9],q[11];
ry(2.312144666165281) q[11];
ry(2.918779764310078) q[13];
cx q[11],q[13];
ry(3.110691954710975) q[11];
ry(-3.0938500656074757) q[13];
cx q[11],q[13];
ry(0.6786343480592939) q[13];
ry(0.8084617954133081) q[15];
cx q[13],q[15];
ry(0.6603923767362905) q[13];
ry(0.8725451853699253) q[15];
cx q[13],q[15];
ry(1.0327871400014796) q[15];
ry(-1.864121408863598) q[17];
cx q[15],q[17];
ry(3.1412440562942643) q[15];
ry(-0.03089399643272012) q[17];
cx q[15],q[17];
ry(1.643061949573017) q[17];
ry(0.5442569958546285) q[19];
cx q[17],q[19];
ry(-0.017165261009242094) q[17];
ry(0.282650866907332) q[19];
cx q[17],q[19];
ry(0.7431537291196895) q[0];
ry(-0.733387108432709) q[1];
cx q[0],q[1];
ry(2.4284288785381083) q[0];
ry(-2.0375028958777994) q[1];
cx q[0],q[1];
ry(-2.7798817859303955) q[2];
ry(-1.0471825390025902) q[3];
cx q[2],q[3];
ry(-1.7847376983531702) q[2];
ry(-3.0233681234537304) q[3];
cx q[2],q[3];
ry(-1.3729225508179501) q[4];
ry(-0.006100033987086917) q[5];
cx q[4],q[5];
ry(0.1548783340402504) q[4];
ry(-2.218677470903086) q[5];
cx q[4],q[5];
ry(-1.0616124118175698) q[6];
ry(0.3795748554758562) q[7];
cx q[6],q[7];
ry(-3.0787851591555433) q[6];
ry(-0.22412207655082295) q[7];
cx q[6],q[7];
ry(-2.876172483689023) q[8];
ry(-1.4738288990366153) q[9];
cx q[8],q[9];
ry(0.7321959495943988) q[8];
ry(-0.13938351715044725) q[9];
cx q[8],q[9];
ry(-0.7395326960665711) q[10];
ry(0.0419573206841148) q[11];
cx q[10],q[11];
ry(-1.0324248269380742) q[10];
ry(-0.5707268545171466) q[11];
cx q[10],q[11];
ry(0.16928477785569904) q[12];
ry(2.9198758439217327) q[13];
cx q[12],q[13];
ry(2.7359356054286916) q[12];
ry(-0.6841298080396029) q[13];
cx q[12],q[13];
ry(2.2802438279488975) q[14];
ry(0.8065688828908142) q[15];
cx q[14],q[15];
ry(-0.2419847567066836) q[14];
ry(3.091158996139547) q[15];
cx q[14],q[15];
ry(-1.2304118285066612) q[16];
ry(-2.4481028970739627) q[17];
cx q[16],q[17];
ry(0.24662977939609387) q[16];
ry(3.107321421598963) q[17];
cx q[16],q[17];
ry(2.1893042636800573) q[18];
ry(-1.387784912464597) q[19];
cx q[18],q[19];
ry(3.0901828182309665) q[18];
ry(2.654076786258863) q[19];
cx q[18],q[19];
ry(-0.5375722753366743) q[0];
ry(2.7916837555339247) q[2];
cx q[0],q[2];
ry(2.161715888192416) q[0];
ry(-2.8160050375775127) q[2];
cx q[0],q[2];
ry(0.5939290051312457) q[2];
ry(2.854083343277233) q[4];
cx q[2],q[4];
ry(-2.8342077719506142) q[2];
ry(-2.9544172833307236) q[4];
cx q[2],q[4];
ry(2.0958804583835056) q[4];
ry(-0.9781027341843398) q[6];
cx q[4],q[6];
ry(-0.566445378571526) q[4];
ry(2.388254537226573) q[6];
cx q[4],q[6];
ry(3.0747134856808223) q[6];
ry(0.9306960498157483) q[8];
cx q[6],q[8];
ry(0.7225933588170967) q[6];
ry(-3.135223574895075) q[8];
cx q[6],q[8];
ry(-1.6599320893447143) q[8];
ry(-0.6343711638686536) q[10];
cx q[8],q[10];
ry(0.04724188832490839) q[8];
ry(2.280963207195117) q[10];
cx q[8],q[10];
ry(-1.144745524605396) q[10];
ry(1.4025595584260717) q[12];
cx q[10],q[12];
ry(-3.139906082390874) q[10];
ry(0.003331304975747962) q[12];
cx q[10],q[12];
ry(-1.7244972401320773) q[12];
ry(2.55239542431209) q[14];
cx q[12],q[14];
ry(0.01884138415774578) q[12];
ry(2.9780637053810652) q[14];
cx q[12],q[14];
ry(2.95115370048074) q[14];
ry(-2.665458181909837) q[16];
cx q[14],q[16];
ry(2.1493792994580287) q[14];
ry(-1.548508795745549) q[16];
cx q[14],q[16];
ry(-1.8827429561270617) q[16];
ry(-1.286993958571171) q[18];
cx q[16],q[18];
ry(0.8005615235253174) q[16];
ry(-1.1158678019764081) q[18];
cx q[16],q[18];
ry(2.160717120457484) q[1];
ry(1.201038965033419) q[3];
cx q[1],q[3];
ry(2.180795751845859) q[1];
ry(-0.11059628096921899) q[3];
cx q[1],q[3];
ry(-1.7832162598598853) q[3];
ry(0.3675619472769469) q[5];
cx q[3],q[5];
ry(-2.9332447960109036) q[3];
ry(0.37435224192994987) q[5];
cx q[3],q[5];
ry(1.4208457343186218) q[5];
ry(-2.3883163257030153) q[7];
cx q[5],q[7];
ry(-2.882526408251954) q[5];
ry(-3.031720915395178) q[7];
cx q[5],q[7];
ry(-1.550482055419504) q[7];
ry(1.0663147494595897) q[9];
cx q[7],q[9];
ry(0.16722152588846484) q[7];
ry(-0.02023276589132047) q[9];
cx q[7],q[9];
ry(-0.9388587255584735) q[9];
ry(-1.180686673085793) q[11];
cx q[9],q[11];
ry(-1.671407277877127) q[9];
ry(-2.0790275338866646) q[11];
cx q[9],q[11];
ry(-0.6656747175281437) q[11];
ry(1.300226440761337) q[13];
cx q[11],q[13];
ry(0.002359574137117626) q[11];
ry(3.1364187463925197) q[13];
cx q[11],q[13];
ry(-0.15111017775097224) q[13];
ry(1.145707196999565) q[15];
cx q[13],q[15];
ry(2.5178267644686896) q[13];
ry(-2.6851602965381747) q[15];
cx q[13],q[15];
ry(-0.7808165421158811) q[15];
ry(-2.073494129866735) q[17];
cx q[15],q[17];
ry(3.1210439601120377) q[15];
ry(0.011362554259495004) q[17];
cx q[15],q[17];
ry(2.9860588359504256) q[17];
ry(2.937278888877187) q[19];
cx q[17],q[19];
ry(0.07427798094894113) q[17];
ry(2.7404491780039346) q[19];
cx q[17],q[19];
ry(-2.4589107068790352) q[0];
ry(-2.42691345817412) q[1];
cx q[0],q[1];
ry(1.3101260844395455) q[0];
ry(0.9322159227298483) q[1];
cx q[0],q[1];
ry(-2.6650357541830383) q[2];
ry(2.5407066208401514) q[3];
cx q[2],q[3];
ry(1.729781025192929) q[2];
ry(-0.6145558258111574) q[3];
cx q[2],q[3];
ry(2.9979904605461156) q[4];
ry(1.2982381236463518) q[5];
cx q[4],q[5];
ry(0.18876968324408286) q[4];
ry(-1.0845467670217506) q[5];
cx q[4],q[5];
ry(1.9603322187474364) q[6];
ry(-2.427590585504534) q[7];
cx q[6],q[7];
ry(-3.044306303189079) q[6];
ry(3.1264747383232505) q[7];
cx q[6],q[7];
ry(-0.016238508434335053) q[8];
ry(0.17723338300909774) q[9];
cx q[8],q[9];
ry(0.017245891021612) q[8];
ry(-1.5030801594006569) q[9];
cx q[8],q[9];
ry(-0.5539396694312472) q[10];
ry(-0.5433672868813032) q[11];
cx q[10],q[11];
ry(-1.4450166371002415) q[10];
ry(-1.9721460315474626) q[11];
cx q[10],q[11];
ry(1.3901436787887684) q[12];
ry(-3.0514215181052724) q[13];
cx q[12],q[13];
ry(-1.4082150822022974) q[12];
ry(2.2188842739726895) q[13];
cx q[12],q[13];
ry(-3.1093529494226115) q[14];
ry(-1.0641028298640334) q[15];
cx q[14],q[15];
ry(1.5566762701363075) q[14];
ry(-1.5092601786160786) q[15];
cx q[14],q[15];
ry(-1.2550200178400257) q[16];
ry(0.11423022029085672) q[17];
cx q[16],q[17];
ry(-2.880011379306197) q[16];
ry(1.9043001825330195) q[17];
cx q[16],q[17];
ry(1.553712733787104) q[18];
ry(1.247861291234907) q[19];
cx q[18],q[19];
ry(2.262774256719081) q[18];
ry(1.187682480093212) q[19];
cx q[18],q[19];
ry(1.3216984735800092) q[0];
ry(1.6259777107993099) q[2];
cx q[0],q[2];
ry(2.8661624174638356) q[0];
ry(-0.41860557730607445) q[2];
cx q[0],q[2];
ry(-0.6008671827736565) q[2];
ry(0.23587225426695735) q[4];
cx q[2],q[4];
ry(3.0316117116088734) q[2];
ry(-0.046997313740450686) q[4];
cx q[2],q[4];
ry(-1.3799245105592384) q[4];
ry(-1.325612373757087) q[6];
cx q[4],q[6];
ry(0.06748357128376803) q[4];
ry(3.12513528650837) q[6];
cx q[4],q[6];
ry(0.4176630797000067) q[6];
ry(-1.2457923494396181) q[8];
cx q[6],q[8];
ry(2.5103751275498936) q[6];
ry(-0.07396228408418626) q[8];
cx q[6],q[8];
ry(2.5811535847041926) q[8];
ry(-1.5749458990379495) q[10];
cx q[8],q[10];
ry(-1.5621693624352089) q[8];
ry(1.3493405294646186) q[10];
cx q[8],q[10];
ry(0.7489431196586871) q[10];
ry(2.6988209908228806) q[12];
cx q[10],q[12];
ry(3.085942292124503) q[10];
ry(-0.10061719157974323) q[12];
cx q[10],q[12];
ry(-1.8914541708451096) q[12];
ry(1.966088308878759) q[14];
cx q[12],q[14];
ry(2.3626521824010336) q[12];
ry(1.1890615112598357) q[14];
cx q[12],q[14];
ry(0.5734825802516081) q[14];
ry(1.174128167082324) q[16];
cx q[14],q[16];
ry(0.001910765289761862) q[14];
ry(-3.1395532453093944) q[16];
cx q[14],q[16];
ry(-0.16695880778511363) q[16];
ry(-0.889268748012376) q[18];
cx q[16],q[18];
ry(-1.4467718639389167) q[16];
ry(-2.1169705706490998) q[18];
cx q[16],q[18];
ry(-0.5416130238362938) q[1];
ry(3.136333455579573) q[3];
cx q[1],q[3];
ry(-0.521248242669301) q[1];
ry(-1.1425524974137797) q[3];
cx q[1],q[3];
ry(-0.5651186367853871) q[3];
ry(1.8472737021321777) q[5];
cx q[3],q[5];
ry(0.2955007742575946) q[3];
ry(1.6419442122979013) q[5];
cx q[3],q[5];
ry(-1.7210764980143065) q[5];
ry(3.0844125877687345) q[7];
cx q[5],q[7];
ry(-2.059780983965692) q[5];
ry(0.01956810396198705) q[7];
cx q[5],q[7];
ry(-0.9641929995535926) q[7];
ry(0.4614146432092818) q[9];
cx q[7],q[9];
ry(0.8089461063111427) q[7];
ry(0.0662510230454112) q[9];
cx q[7],q[9];
ry(-2.252221124787577) q[9];
ry(-1.269978581521907) q[11];
cx q[9],q[11];
ry(2.6850678337738385) q[9];
ry(0.22313219873913478) q[11];
cx q[9],q[11];
ry(1.7744979125468676) q[11];
ry(0.9775504213508484) q[13];
cx q[11],q[13];
ry(-2.972044632349847) q[11];
ry(-3.122409065506591) q[13];
cx q[11],q[13];
ry(1.4484989364394782) q[13];
ry(0.8475205841239888) q[15];
cx q[13],q[15];
ry(-3.0157648173756484) q[13];
ry(1.014465203982236) q[15];
cx q[13],q[15];
ry(-1.3704765598278792) q[15];
ry(0.545977276971411) q[17];
cx q[15],q[17];
ry(3.1167655274628103) q[15];
ry(-3.08439330897841) q[17];
cx q[15],q[17];
ry(0.461574104821735) q[17];
ry(1.274985785406771) q[19];
cx q[17],q[19];
ry(-1.1234797931966547) q[17];
ry(2.9648171764772475) q[19];
cx q[17],q[19];
ry(-1.9543751272868413) q[0];
ry(0.943778273780693) q[1];
cx q[0],q[1];
ry(1.9151301014245705) q[0];
ry(-0.17205489997924642) q[1];
cx q[0],q[1];
ry(-3.023088373228783) q[2];
ry(-2.564779845466741) q[3];
cx q[2],q[3];
ry(1.4332244684070392) q[2];
ry(1.3550391659439782) q[3];
cx q[2],q[3];
ry(1.5178793317737265) q[4];
ry(0.8857821473203114) q[5];
cx q[4],q[5];
ry(3.1005252593517634) q[4];
ry(1.29574023646403) q[5];
cx q[4],q[5];
ry(1.244505654414179) q[6];
ry(-2.8543388052616043) q[7];
cx q[6],q[7];
ry(-3.1132046953926586) q[6];
ry(3.035494551217832) q[7];
cx q[6],q[7];
ry(1.1739589990663433) q[8];
ry(1.5613727559472583) q[9];
cx q[8],q[9];
ry(3.1136068469432523) q[8];
ry(3.141130966803701) q[9];
cx q[8],q[9];
ry(2.500983766015228) q[10];
ry(2.7547567351378994) q[11];
cx q[10],q[11];
ry(-0.0085649653209634) q[10];
ry(-0.02166648895003931) q[11];
cx q[10],q[11];
ry(0.9585836756819894) q[12];
ry(-1.0193590395114764) q[13];
cx q[12],q[13];
ry(0.06252607229460884) q[12];
ry(-0.0770391137411286) q[13];
cx q[12],q[13];
ry(1.6287644310305875) q[14];
ry(-2.8941364407066694) q[15];
cx q[14],q[15];
ry(-2.9962488831743563) q[14];
ry(-0.2042699382865145) q[15];
cx q[14],q[15];
ry(1.5659646625329682) q[16];
ry(1.6050857276560349) q[17];
cx q[16],q[17];
ry(2.8246111850483793) q[16];
ry(1.7270571463793265) q[17];
cx q[16],q[17];
ry(0.08935010973267757) q[18];
ry(0.5476917497316798) q[19];
cx q[18],q[19];
ry(-0.4440607269772939) q[18];
ry(2.335437232679237) q[19];
cx q[18],q[19];
ry(2.0147395495059683) q[0];
ry(-0.30586500039855924) q[2];
cx q[0],q[2];
ry(2.8227282867560755) q[0];
ry(2.398910177003467) q[2];
cx q[0],q[2];
ry(-0.13775502162154354) q[2];
ry(-0.5234768955277218) q[4];
cx q[2],q[4];
ry(-3.106534484841695) q[2];
ry(-0.389534949593691) q[4];
cx q[2],q[4];
ry(-0.6640738097187128) q[4];
ry(-2.833366372755703) q[6];
cx q[4],q[6];
ry(-2.364787699098248) q[4];
ry(-0.24294047397292237) q[6];
cx q[4],q[6];
ry(-1.5586979805508316) q[6];
ry(2.5678865490212472) q[8];
cx q[6],q[8];
ry(-0.527731848082941) q[6];
ry(-3.1079346981505673) q[8];
cx q[6],q[8];
ry(1.2751313925595176) q[8];
ry(-2.6568998074603325) q[10];
cx q[8],q[10];
ry(-0.9533562502670129) q[8];
ry(0.2712633965004929) q[10];
cx q[8],q[10];
ry(3.0562722287863218) q[10];
ry(-0.3075214593033004) q[12];
cx q[10],q[12];
ry(0.12395254089779684) q[10];
ry(-3.081165272693045) q[12];
cx q[10],q[12];
ry(1.5024291538034884) q[12];
ry(2.778940119612516) q[14];
cx q[12],q[14];
ry(1.497243764975886) q[12];
ry(-1.7160656930979847) q[14];
cx q[12],q[14];
ry(1.3938655869542858) q[14];
ry(1.9362041215779904) q[16];
cx q[14],q[16];
ry(-0.08833311186038659) q[14];
ry(-0.015637812321164617) q[16];
cx q[14],q[16];
ry(2.909467612956717) q[16];
ry(-1.587912008419062) q[18];
cx q[16],q[18];
ry(0.8410775475490517) q[16];
ry(-1.135559593764129) q[18];
cx q[16],q[18];
ry(-1.1961243421462369) q[1];
ry(1.9104470382302075) q[3];
cx q[1],q[3];
ry(-0.15232549731784673) q[1];
ry(2.85070290159607) q[3];
cx q[1],q[3];
ry(-2.583043159346517) q[3];
ry(-0.07833502796222319) q[5];
cx q[3],q[5];
ry(-3.138414413687246) q[3];
ry(-1.7315808901814407) q[5];
cx q[3],q[5];
ry(-0.865214127921309) q[5];
ry(-0.5300273190612703) q[7];
cx q[5],q[7];
ry(0.6906248444687156) q[5];
ry(2.7112158982366625) q[7];
cx q[5],q[7];
ry(1.5925854021168098) q[7];
ry(2.427522726241399) q[9];
cx q[7],q[9];
ry(0.906229777070119) q[7];
ry(-2.2494297152285103) q[9];
cx q[7],q[9];
ry(-1.652830118184544) q[9];
ry(-2.373438740903725) q[11];
cx q[9],q[11];
ry(-0.8467687131762958) q[9];
ry(-1.5701602523903553) q[11];
cx q[9],q[11];
ry(2.3534777762596706) q[11];
ry(-0.4961493184570244) q[13];
cx q[11],q[13];
ry(-2.3807793112915783) q[11];
ry(0.10449747887211928) q[13];
cx q[11],q[13];
ry(0.40985808936254475) q[13];
ry(-2.4774203239998887) q[15];
cx q[13],q[15];
ry(0.6540767611024032) q[13];
ry(-1.9495850127289396) q[15];
cx q[13],q[15];
ry(0.17839878985496255) q[15];
ry(1.0370392468013483) q[17];
cx q[15],q[17];
ry(3.1408884497346707) q[15];
ry(-3.4888730086279907e-05) q[17];
cx q[15],q[17];
ry(0.0406196293762203) q[17];
ry(1.0872938606297073) q[19];
cx q[17],q[19];
ry(1.428170422579421) q[17];
ry(-0.09992941201873844) q[19];
cx q[17],q[19];
ry(-1.6057643301128899) q[0];
ry(0.13731068790417886) q[1];
cx q[0],q[1];
ry(2.752385577299103) q[0];
ry(0.28972547317120645) q[1];
cx q[0],q[1];
ry(-1.231973391372625) q[2];
ry(-0.9218482330630078) q[3];
cx q[2],q[3];
ry(-0.8021291935882209) q[2];
ry(2.195261678231004) q[3];
cx q[2],q[3];
ry(1.0417814870146307) q[4];
ry(2.8493294997008802) q[5];
cx q[4],q[5];
ry(0.014322775564509789) q[4];
ry(0.05022831934010325) q[5];
cx q[4],q[5];
ry(-2.712253609550925) q[6];
ry(-0.7187899779023174) q[7];
cx q[6],q[7];
ry(-3.113497809103624) q[6];
ry(0.007534298251334449) q[7];
cx q[6],q[7];
ry(-3.1254289752672926) q[8];
ry(-2.3641540867369994) q[9];
cx q[8],q[9];
ry(-3.13633234458569) q[8];
ry(-3.122550790402715) q[9];
cx q[8],q[9];
ry(0.8208698394763253) q[10];
ry(0.4752571990677179) q[11];
cx q[10],q[11];
ry(0.01870323847916312) q[10];
ry(3.0963136446579056) q[11];
cx q[10],q[11];
ry(-1.479393663452007) q[12];
ry(-2.6033521058307323) q[13];
cx q[12],q[13];
ry(0.03701357974031527) q[12];
ry(-0.305369027447572) q[13];
cx q[12],q[13];
ry(-1.6552916020854815) q[14];
ry(-0.6696595510759213) q[15];
cx q[14],q[15];
ry(3.1116775768435807) q[14];
ry(-3.1404557324150897) q[15];
cx q[14],q[15];
ry(-2.1244277725262117) q[16];
ry(-3.1263662126643275) q[17];
cx q[16],q[17];
ry(0.9436394309735535) q[16];
ry(1.4953941513158604) q[17];
cx q[16],q[17];
ry(-2.3518005993740667) q[18];
ry(-1.2444903410865715) q[19];
cx q[18],q[19];
ry(-1.075523223922186) q[18];
ry(1.4301225077589634) q[19];
cx q[18],q[19];
ry(2.1331577851087458) q[0];
ry(-2.0680935445302717) q[2];
cx q[0],q[2];
ry(-2.8805911273872518) q[0];
ry(1.2165725227575344) q[2];
cx q[0],q[2];
ry(2.869787246789242) q[2];
ry(1.96033226918491) q[4];
cx q[2],q[4];
ry(3.127070768237583) q[2];
ry(2.954491305791242) q[4];
cx q[2],q[4];
ry(-1.1743790290339762) q[4];
ry(1.8369340391731899) q[6];
cx q[4],q[6];
ry(-2.8165299321429744) q[4];
ry(0.8664706978169996) q[6];
cx q[4],q[6];
ry(1.2106134299075306) q[6];
ry(2.2325754076400823) q[8];
cx q[6],q[8];
ry(0.420469846774012) q[6];
ry(0.5089974886275123) q[8];
cx q[6],q[8];
ry(-0.8959779814083526) q[8];
ry(0.5034167569508936) q[10];
cx q[8],q[10];
ry(1.957165751785472) q[8];
ry(-0.8152836980418731) q[10];
cx q[8],q[10];
ry(2.5074574230573354) q[10];
ry(3.0011594753456747) q[12];
cx q[10],q[12];
ry(2.381891006433037) q[10];
ry(0.07418672966463007) q[12];
cx q[10],q[12];
ry(1.3628547514735043) q[12];
ry(-2.016901599485549) q[14];
cx q[12],q[14];
ry(-2.741921343804536) q[12];
ry(-2.389145782732638) q[14];
cx q[12],q[14];
ry(-2.703735196635406) q[14];
ry(0.08858442109394471) q[16];
cx q[14],q[16];
ry(3.11662973355016) q[14];
ry(0.1229025565130905) q[16];
cx q[14],q[16];
ry(-2.8686484812101583) q[16];
ry(1.903702424507662) q[18];
cx q[16],q[18];
ry(1.0506077696825478) q[16];
ry(0.2262188739404678) q[18];
cx q[16],q[18];
ry(2.2997676783287195) q[1];
ry(-2.8047541033440995) q[3];
cx q[1],q[3];
ry(0.11413224532169863) q[1];
ry(-0.42351929384702697) q[3];
cx q[1],q[3];
ry(2.5676245410263743) q[3];
ry(0.03773799113415753) q[5];
cx q[3],q[5];
ry(0.004095872005134282) q[3];
ry(3.1339823051431392) q[5];
cx q[3],q[5];
ry(-0.5775217665845833) q[5];
ry(1.0587156792698096) q[7];
cx q[5],q[7];
ry(0.5262810729414067) q[5];
ry(-3.079432219173529) q[7];
cx q[5],q[7];
ry(-0.10107769543751033) q[7];
ry(-2.2179164840500247) q[9];
cx q[7],q[9];
ry(-0.13927297608796027) q[7];
ry(-0.7662138943835197) q[9];
cx q[7],q[9];
ry(-0.7383148488539494) q[9];
ry(-0.07702557127784004) q[11];
cx q[9],q[11];
ry(-2.6251193086008) q[9];
ry(0.1045601223947017) q[11];
cx q[9],q[11];
ry(1.0743839326446634) q[11];
ry(-0.02712676049379148) q[13];
cx q[11],q[13];
ry(2.492291528069929) q[11];
ry(0.05960060162069514) q[13];
cx q[11],q[13];
ry(2.16794863851492) q[13];
ry(1.7188204268729317) q[15];
cx q[13],q[15];
ry(-2.3898526879865267) q[13];
ry(-1.9007869805247413) q[15];
cx q[13],q[15];
ry(1.8778959687561798) q[15];
ry(0.788744129709595) q[17];
cx q[15],q[17];
ry(3.1373539878236403) q[15];
ry(3.1394734289695667) q[17];
cx q[15],q[17];
ry(2.896752545118663) q[17];
ry(1.852635908059322) q[19];
cx q[17],q[19];
ry(-1.4021845351518465) q[17];
ry(-0.300263782167681) q[19];
cx q[17],q[19];
ry(0.6899901187099086) q[0];
ry(2.652785269731507) q[1];
cx q[0],q[1];
ry(1.7265364339960987) q[0];
ry(-1.6402556559933572) q[1];
cx q[0],q[1];
ry(1.1276632916188394) q[2];
ry(-2.752480536150047) q[3];
cx q[2],q[3];
ry(-2.4642924968481488) q[2];
ry(-1.9944645165896038) q[3];
cx q[2],q[3];
ry(0.021323710126329942) q[4];
ry(0.82910817955422) q[5];
cx q[4],q[5];
ry(-3.137191738508558) q[4];
ry(-3.0700713709646323) q[5];
cx q[4],q[5];
ry(2.0190263732672102) q[6];
ry(-2.9390342535991243) q[7];
cx q[6],q[7];
ry(3.1318844668621364) q[6];
ry(-0.02543072950313441) q[7];
cx q[6],q[7];
ry(-1.5580567160370435) q[8];
ry(-0.8795646468252878) q[9];
cx q[8],q[9];
ry(-3.1332708171668466) q[8];
ry(-3.132330588160199) q[9];
cx q[8],q[9];
ry(-1.5971843415351703) q[10];
ry(-1.6457003418158471) q[11];
cx q[10],q[11];
ry(3.1364750877366636) q[10];
ry(0.0006251466913155851) q[11];
cx q[10],q[11];
ry(-0.7924060341100247) q[12];
ry(2.587143870898621) q[13];
cx q[12],q[13];
ry(-0.026821596827409935) q[12];
ry(-3.135303287363386) q[13];
cx q[12],q[13];
ry(-3.008655749889675) q[14];
ry(-1.4268045162484624) q[15];
cx q[14],q[15];
ry(3.113381107525787) q[14];
ry(-0.018341228676012712) q[15];
cx q[14],q[15];
ry(-1.0183184346221648) q[16];
ry(-0.2569668930702434) q[17];
cx q[16],q[17];
ry(2.893085996791825) q[16];
ry(3.0657818358927886) q[17];
cx q[16],q[17];
ry(-1.614616880728687) q[18];
ry(-0.9294550105502886) q[19];
cx q[18],q[19];
ry(-1.3804290948495663) q[18];
ry(-2.110970294321158) q[19];
cx q[18],q[19];
ry(0.7255044416162475) q[0];
ry(-0.5751284002204472) q[2];
cx q[0],q[2];
ry(2.9490704798133067) q[0];
ry(-0.1613204348344297) q[2];
cx q[0],q[2];
ry(2.1533962883086986) q[2];
ry(-0.16028041969162463) q[4];
cx q[2],q[4];
ry(-3.100180204684714) q[2];
ry(3.0976916378916934) q[4];
cx q[2],q[4];
ry(-0.8692053824419678) q[4];
ry(-0.45375950372621693) q[6];
cx q[4],q[6];
ry(2.935079128055445) q[4];
ry(0.1526198932397668) q[6];
cx q[4],q[6];
ry(2.9909703681415793) q[6];
ry(0.8963775292871947) q[8];
cx q[6],q[8];
ry(2.6935363177453526) q[6];
ry(1.4942885683433815) q[8];
cx q[6],q[8];
ry(0.5646275729876651) q[8];
ry(0.8939011758258016) q[10];
cx q[8],q[10];
ry(-0.06262057153658948) q[8];
ry(0.3642945550053498) q[10];
cx q[8],q[10];
ry(-3.0824030879139266) q[10];
ry(2.6089091516361216) q[12];
cx q[10],q[12];
ry(-1.6598895419730724) q[10];
ry(-1.6454980802672958) q[12];
cx q[10],q[12];
ry(-0.14453545952403293) q[12];
ry(-0.11858455747360795) q[14];
cx q[12],q[14];
ry(-3.131910969174999) q[12];
ry(2.5011506002468487) q[14];
cx q[12],q[14];
ry(-1.3018442200980895) q[14];
ry(2.896455735139507) q[16];
cx q[14],q[16];
ry(0.1435479106828518) q[14];
ry(-1.327445161973043) q[16];
cx q[14],q[16];
ry(-2.6603315287973146) q[16];
ry(2.3747030299692495) q[18];
cx q[16],q[18];
ry(0.17358238345754537) q[16];
ry(-0.3227113126057478) q[18];
cx q[16],q[18];
ry(-0.2066277217065462) q[1];
ry(-0.2380128376443023) q[3];
cx q[1],q[3];
ry(-1.1491030602569063) q[1];
ry(3.0172483914434003) q[3];
cx q[1],q[3];
ry(2.383311263900863) q[3];
ry(3.1056944242125692) q[5];
cx q[3],q[5];
ry(-2.0690156716775534) q[3];
ry(-3.093166040836748) q[5];
cx q[3],q[5];
ry(-0.6662779187916588) q[5];
ry(-1.2249034916203554) q[7];
cx q[5],q[7];
ry(-3.066011989057693) q[5];
ry(-0.06395132038250573) q[7];
cx q[5],q[7];
ry(3.033423486834214) q[7];
ry(-1.330920216802015) q[9];
cx q[7],q[9];
ry(-3.0833968207399702) q[7];
ry(1.6122602443186302) q[9];
cx q[7],q[9];
ry(-0.6420436024125556) q[9];
ry(1.5030281563312684) q[11];
cx q[9],q[11];
ry(-0.1517237303061111) q[9];
ry(2.747554829790776) q[11];
cx q[9],q[11];
ry(-0.23048511882671632) q[11];
ry(-0.07773061864203079) q[13];
cx q[11],q[13];
ry(0.34474437866698193) q[11];
ry(2.436954524994903) q[13];
cx q[11],q[13];
ry(2.911558279103237) q[13];
ry(2.436999570546072) q[15];
cx q[13],q[15];
ry(2.297224310601482) q[13];
ry(-1.6819635832408801) q[15];
cx q[13],q[15];
ry(-0.4757649178350789) q[15];
ry(2.647393201503004) q[17];
cx q[15],q[17];
ry(-0.03503100127841289) q[15];
ry(-2.266852684622596) q[17];
cx q[15],q[17];
ry(-1.6343742327127928) q[17];
ry(-0.705314010902219) q[19];
cx q[17],q[19];
ry(0.39644621356235193) q[17];
ry(2.7224581395256924) q[19];
cx q[17],q[19];
ry(-1.6155339067145853) q[0];
ry(0.6814570736721981) q[1];
cx q[0],q[1];
ry(0.21857012917195462) q[0];
ry(-2.2418166934349397) q[1];
cx q[0],q[1];
ry(0.8662925757510344) q[2];
ry(-2.908158804200646) q[3];
cx q[2],q[3];
ry(0.02187574022082117) q[2];
ry(1.153507952042126) q[3];
cx q[2],q[3];
ry(0.35865008730071357) q[4];
ry(-2.187225241516831) q[5];
cx q[4],q[5];
ry(-0.01853455243736288) q[4];
ry(-0.023474933139097764) q[5];
cx q[4],q[5];
ry(1.872317919170408) q[6];
ry(-2.5719440650893435) q[7];
cx q[6],q[7];
ry(-0.021202637928986023) q[6];
ry(-3.1391291747966763) q[7];
cx q[6],q[7];
ry(-0.2351222151700082) q[8];
ry(2.6394370348745073) q[9];
cx q[8],q[9];
ry(-3.138010991980389) q[8];
ry(-3.141164663109815) q[9];
cx q[8],q[9];
ry(-2.2282446266649267) q[10];
ry(1.4052849646737287) q[11];
cx q[10],q[11];
ry(3.1340120729821384) q[10];
ry(-0.012750514673292024) q[11];
cx q[10],q[11];
ry(-2.3704802347875247) q[12];
ry(2.352238102715031) q[13];
cx q[12],q[13];
ry(3.1411348151740337) q[12];
ry(-0.003599235117202504) q[13];
cx q[12],q[13];
ry(2.8657757408803963) q[14];
ry(-1.345690328874035) q[15];
cx q[14],q[15];
ry(-3.139326979010504) q[14];
ry(3.120170657693184) q[15];
cx q[14],q[15];
ry(-1.3861165428250228) q[16];
ry(-1.554028826960686) q[17];
cx q[16],q[17];
ry(0.05356212223882959) q[16];
ry(0.00576349276722965) q[17];
cx q[16],q[17];
ry(2.3750216911680804) q[18];
ry(2.9598776726222473) q[19];
cx q[18],q[19];
ry(-0.024469661450147458) q[18];
ry(3.0256725128450705) q[19];
cx q[18],q[19];
ry(2.1177653721234524) q[0];
ry(0.23690553986168175) q[2];
cx q[0],q[2];
ry(1.1496013349554373) q[0];
ry(1.7647822999536564) q[2];
cx q[0],q[2];
ry(0.5366592784830463) q[2];
ry(-1.6703369150947915) q[4];
cx q[2],q[4];
ry(0.027080220742804525) q[2];
ry(2.677327873072631) q[4];
cx q[2],q[4];
ry(2.917006560782123) q[4];
ry(-0.9523860193325273) q[6];
cx q[4],q[6];
ry(3.133402164370437) q[4];
ry(-3.031252624037153) q[6];
cx q[4],q[6];
ry(0.8352416435712734) q[6];
ry(-2.703341397394841) q[8];
cx q[6],q[8];
ry(-0.5173104068173286) q[6];
ry(-1.3979472388883631) q[8];
cx q[6],q[8];
ry(0.21391082282894916) q[8];
ry(-2.572940736504437) q[10];
cx q[8],q[10];
ry(-2.905493646681111) q[8];
ry(-0.08911423976086397) q[10];
cx q[8],q[10];
ry(-1.8063158943313182) q[10];
ry(-2.9627081491889133) q[12];
cx q[10],q[12];
ry(1.4080029731431674) q[10];
ry(-1.5022929266787328) q[12];
cx q[10],q[12];
ry(0.4654028004446422) q[12];
ry(-1.8054184577877477) q[14];
cx q[12],q[14];
ry(-1.04886338232601) q[12];
ry(3.1391710677017484) q[14];
cx q[12],q[14];
ry(-2.060171267340741) q[14];
ry(1.4221677931820103) q[16];
cx q[14],q[16];
ry(-2.708474393101252) q[14];
ry(1.3817970385357086) q[16];
cx q[14],q[16];
ry(1.6500145676408282) q[16];
ry(3.0922861263309898) q[18];
cx q[16],q[18];
ry(1.558384874913327) q[16];
ry(-1.5803681948459238) q[18];
cx q[16],q[18];
ry(-2.826373982275175) q[1];
ry(-1.9503153706874015) q[3];
cx q[1],q[3];
ry(-0.5591086693082844) q[1];
ry(2.868519199548392) q[3];
cx q[1],q[3];
ry(0.21238186159726044) q[3];
ry(2.2488768674016932) q[5];
cx q[3],q[5];
ry(-2.0765679027214556) q[3];
ry(1.5459413123910004) q[5];
cx q[3],q[5];
ry(-1.3915812944341377) q[5];
ry(-2.897570366289079) q[7];
cx q[5],q[7];
ry(-1.008369738322763) q[5];
ry(0.1552938158284345) q[7];
cx q[5],q[7];
ry(-1.6083819959907215) q[7];
ry(-1.4411597923949904) q[9];
cx q[7],q[9];
ry(-1.5976724989001874) q[7];
ry(1.665524878146499) q[9];
cx q[7],q[9];
ry(-2.469989938438425) q[9];
ry(-0.11962876482490381) q[11];
cx q[9],q[11];
ry(-1.626906988253059) q[9];
ry(-3.126001059900314) q[11];
cx q[9],q[11];
ry(-2.1490637821073397) q[11];
ry(0.09075852972585263) q[13];
cx q[11],q[13];
ry(1.5649422527985561) q[11];
ry(-3.135778361918507) q[13];
cx q[11],q[13];
ry(-1.5370703515900113) q[13];
ry(0.22607952523510016) q[15];
cx q[13],q[15];
ry(1.5742826221752262) q[13];
ry(3.134847963485707) q[15];
cx q[13],q[15];
ry(1.0609889126812795) q[15];
ry(1.5690509100902688) q[17];
cx q[15],q[17];
ry(-2.7528926581200044) q[15];
ry(-2.3004404881159477) q[17];
cx q[15],q[17];
ry(-1.5676728216646199) q[17];
ry(-1.6640632722839652) q[19];
cx q[17],q[19];
ry(1.5711497395555254) q[17];
ry(-1.5704068139575575) q[19];
cx q[17],q[19];
ry(-1.1974520215365887) q[0];
ry(-2.7234000850086586) q[1];
ry(-0.423177572023864) q[2];
ry(0.07298265525646563) q[3];
ry(-1.9286088867079263) q[4];
ry(2.891653108015511) q[5];
ry(-1.9601727040370935) q[6];
ry(2.985059199637637) q[7];
ry(-3.064623852407134) q[8];
ry(2.057840067903036) q[9];
ry(-2.9284160402139676) q[10];
ry(-0.6963553602772166) q[11];
ry(2.135449699021505) q[12];
ry(-0.08605378377400719) q[13];
ry(-1.1434507098684188) q[14];
ry(-1.9353468201571227) q[15];
ry(0.23799204780192648) q[16];
ry(3.01075483138781) q[17];
ry(-1.4783617258675372) q[18];
ry(-1.9801956672770156) q[19];