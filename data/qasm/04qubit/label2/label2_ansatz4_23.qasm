OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.0746630679230016) q[0];
rz(-2.0739122249266564) q[0];
ry(0.6159455104920077) q[1];
rz(-0.03671250724316006) q[1];
ry(-0.6612102348020188) q[2];
rz(-3.1001846265717075) q[2];
ry(-2.828013172754217) q[3];
rz(2.293704714246587) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.7896362556222238) q[0];
rz(2.4946214727181797) q[0];
ry(-0.004869925990569791) q[1];
rz(1.8296927071493077) q[1];
ry(1.7314304637734264) q[2];
rz(1.725024710050264) q[2];
ry(-1.5328866766119562) q[3];
rz(0.12653092509948308) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.457860095455428) q[0];
rz(0.946605687755013) q[0];
ry(1.961432211404027) q[1];
rz(2.2383986906616276) q[1];
ry(-1.6534493337972975) q[2];
rz(2.6690245955027323) q[2];
ry(-2.826579157750494) q[3];
rz(2.047050099920443) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.9514237052595993) q[0];
rz(0.28173226696500026) q[0];
ry(1.6908718822044035) q[1];
rz(-0.02281049674233238) q[1];
ry(1.050133421693113) q[2];
rz(3.140026133891194) q[2];
ry(2.88605596787221) q[3];
rz(0.9088711020996717) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.154944015336214) q[0];
rz(1.7840854301400353) q[0];
ry(2.3843410432793912) q[1];
rz(0.4154151828548232) q[1];
ry(-0.021520584178244193) q[2];
rz(-0.6612399311012123) q[2];
ry(1.761691201342321) q[3];
rz(-0.6051655582203979) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.020596329620508703) q[0];
rz(-1.3838351670114062) q[0];
ry(1.2686218692767859) q[1];
rz(-2.431805734167476) q[1];
ry(0.5036848510343194) q[2];
rz(-2.961153580811712) q[2];
ry(-1.1657690040295925) q[3];
rz(-0.9718147180008776) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.8922748065140675) q[0];
rz(2.7328909756235453) q[0];
ry(2.6722562290420178) q[1];
rz(0.1263653912076402) q[1];
ry(1.995260767925639) q[2];
rz(-1.5715263478460821) q[2];
ry(0.02082578411724868) q[3];
rz(1.7157744695370372) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.704408202603524) q[0];
rz(2.705969852079507) q[0];
ry(-2.088447986437571) q[1];
rz(0.010706260292709045) q[1];
ry(-1.2712651932999046) q[2];
rz(-0.619127197120971) q[2];
ry(-0.45676340657120473) q[3];
rz(-0.9174064184701354) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.0730850252023183) q[0];
rz(-2.018480469100213) q[0];
ry(-2.0567943505767783) q[1];
rz(-0.7819438622568751) q[1];
ry(2.352635952230363) q[2];
rz(-3.057622735444182) q[2];
ry(1.1399962846759812) q[3];
rz(-0.4081826128847511) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.2975261506340736) q[0];
rz(-1.9189700926553428) q[0];
ry(-2.379157448978544) q[1];
rz(-2.1961560615702274) q[1];
ry(2.7113844468934545) q[2];
rz(0.025582361034286905) q[2];
ry(2.5779084676882618) q[3];
rz(0.2933857756486545) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.217543212300414) q[0];
rz(1.3550498580732837) q[0];
ry(-1.4650718104770828) q[1];
rz(-0.6351062636134897) q[1];
ry(-0.8366094230712999) q[2];
rz(2.567703398755936) q[2];
ry(1.2831893932084513) q[3];
rz(1.3105609863508239) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.18766606391199403) q[0];
rz(-0.08388830981538761) q[0];
ry(0.02618128944733032) q[1];
rz(2.452949788491384) q[1];
ry(-1.3497938914255387) q[2];
rz(-2.916779343420083) q[2];
ry(-2.2549785945140624) q[3];
rz(0.37484119961212187) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.198629585604338) q[0];
rz(2.0543488771327683) q[0];
ry(0.2810948148796504) q[1];
rz(0.5851202198483604) q[1];
ry(-1.7243309557895947) q[2];
rz(-0.9160172871906536) q[2];
ry(-3.0092950761315476) q[3];
rz(-2.4789082448821804) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7632011098519689) q[0];
rz(1.0245163950117444) q[0];
ry(-0.8887690126920562) q[1];
rz(-1.6196736594202805) q[1];
ry(2.4559168756704484) q[2];
rz(-0.5359620388845202) q[2];
ry(-1.2790289006530922) q[3];
rz(1.3181434903375073) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.4207660514957325) q[0];
rz(1.85374604253989) q[0];
ry(-0.560214859037398) q[1];
rz(-1.9710903297329976) q[1];
ry(0.18362458813490698) q[2];
rz(0.05477991494297229) q[2];
ry(-0.3114616266693914) q[3];
rz(2.6341547573058564) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.884637995382261) q[0];
rz(-1.8132862140430215) q[0];
ry(0.005721629275152758) q[1];
rz(2.010175806081386) q[1];
ry(-0.09297264137862063) q[2];
rz(2.2465691788060416) q[2];
ry(1.175506212438541) q[3];
rz(-1.0032827838938745) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.4225054637705434) q[0];
rz(-1.5791029562421937) q[0];
ry(-2.915556440506208) q[1];
rz(3.0736638703817785) q[1];
ry(1.76878566332379) q[2];
rz(1.8771245860203631) q[2];
ry(0.40264901691981514) q[3];
rz(0.518482643253107) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.9456208344645014) q[0];
rz(0.09434166488979566) q[0];
ry(-0.9355033961733188) q[1];
rz(-0.17891419874189185) q[1];
ry(1.6362129263220637) q[2];
rz(-0.255207735199848) q[2];
ry(2.20641897041219) q[3];
rz(-1.7556671927674046) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.4355520067289902) q[0];
rz(1.444147756319257) q[0];
ry(-0.1335644912962115) q[1];
rz(-0.11155201618738085) q[1];
ry(0.5422784806657904) q[2];
rz(-2.490759580374576) q[2];
ry(-0.22507199309979775) q[3];
rz(-1.9359632964049673) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.5938646374827918) q[0];
rz(-2.6433770710545486) q[0];
ry(0.8788499392324799) q[1];
rz(-2.3737850935737623) q[1];
ry(-0.5253124656881427) q[2];
rz(-1.6795791171903898) q[2];
ry(-0.48277752534677687) q[3];
rz(-0.43062013598348514) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.952628683473729) q[0];
rz(2.475988350877922) q[0];
ry(-1.413313044695336) q[1];
rz(-1.1868767025024294) q[1];
ry(0.6081191051555901) q[2];
rz(2.353532311835709) q[2];
ry(-0.6847147979911377) q[3];
rz(-0.5111436426622403) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.943442830692195) q[0];
rz(-1.2801244402045817) q[0];
ry(-0.22576944190159282) q[1];
rz(2.627847037148236) q[1];
ry(0.07505247431619641) q[2];
rz(-1.5121475038395618) q[2];
ry(-0.5290908098280686) q[3];
rz(0.7554416393320169) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.18800410193085) q[0];
rz(-2.3886469794263046) q[0];
ry(-1.9254938095387106) q[1];
rz(-1.690530089026173) q[1];
ry(0.2541135990552714) q[2];
rz(0.3362334287479527) q[2];
ry(2.2461807785689305) q[3];
rz(-1.7540174657863217) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.9706593187330776) q[0];
rz(2.4716110096777597) q[0];
ry(1.1971132782042817) q[1];
rz(-1.38663299466666) q[1];
ry(-0.7313711786509155) q[2];
rz(0.04324354708007494) q[2];
ry(-2.34167053153278) q[3];
rz(1.726836046372842) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.8634225500218706) q[0];
rz(-0.550399231575812) q[0];
ry(-3.1163510874764455) q[1];
rz(-1.320489231548054) q[1];
ry(3.1295214043165958) q[2];
rz(-0.14702782303120648) q[2];
ry(-0.9545567546656892) q[3];
rz(-1.608807515898663) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.9301975178883968) q[0];
rz(1.3946267974205675) q[0];
ry(-0.34803639099877937) q[1];
rz(-0.7040739309038099) q[1];
ry(0.4811600499133002) q[2];
rz(2.6386300347875884) q[2];
ry(-0.3139122885466632) q[3];
rz(-2.836301399742064) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.702547968765046) q[0];
rz(-0.09212473172295138) q[0];
ry(1.6393113378327318) q[1];
rz(-2.292649257729268) q[1];
ry(-0.49691758727225616) q[2];
rz(0.7812349471865486) q[2];
ry(-1.0789512585674546) q[3];
rz(3.0090812126752846) q[3];