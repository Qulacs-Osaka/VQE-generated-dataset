OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.9309295393999877) q[0];
ry(1.0445679751765828) q[1];
cx q[0],q[1];
ry(-1.5558824193875949) q[0];
ry(2.0900156078530463) q[1];
cx q[0],q[1];
ry(1.0579645391396806) q[1];
ry(-2.2815678555004757) q[2];
cx q[1],q[2];
ry(0.012414432388298854) q[1];
ry(-0.521124933641024) q[2];
cx q[1],q[2];
ry(-0.33012803719717887) q[2];
ry(2.9362179954844287) q[3];
cx q[2],q[3];
ry(0.8834491007223045) q[2];
ry(1.6717386994124181) q[3];
cx q[2],q[3];
ry(2.5174227143323233) q[3];
ry(-0.18074697487468327) q[4];
cx q[3],q[4];
ry(0.3905796939095847) q[3];
ry(0.009372129224069424) q[4];
cx q[3],q[4];
ry(-0.1565379875761532) q[4];
ry(1.6831202843396582) q[5];
cx q[4],q[5];
ry(-2.2127806300566775) q[4];
ry(-0.4051533825972591) q[5];
cx q[4],q[5];
ry(2.1589406067291037) q[5];
ry(-2.066942071613682) q[6];
cx q[5],q[6];
ry(1.6657807689815005) q[5];
ry(0.1936705029140704) q[6];
cx q[5],q[6];
ry(2.1650995031942393) q[6];
ry(1.2461750423082127) q[7];
cx q[6],q[7];
ry(-2.6816716833298404) q[6];
ry(-0.5470927155442405) q[7];
cx q[6],q[7];
ry(0.5214425535850548) q[7];
ry(1.2071899617960922) q[8];
cx q[7],q[8];
ry(-2.5297855115171113) q[7];
ry(0.42002297290955787) q[8];
cx q[7],q[8];
ry(-1.8389142662814226) q[8];
ry(2.4773471106628433) q[9];
cx q[8],q[9];
ry(-1.9547438392482313) q[8];
ry(2.1456905020915014) q[9];
cx q[8],q[9];
ry(-1.4837286000637737) q[9];
ry(2.278585515956875) q[10];
cx q[9],q[10];
ry(-2.92967666885997) q[9];
ry(1.344413339968054) q[10];
cx q[9],q[10];
ry(-1.3250866761457862) q[10];
ry(-2.503529456849435) q[11];
cx q[10],q[11];
ry(-2.595389735977296) q[10];
ry(2.663176078241517) q[11];
cx q[10],q[11];
ry(1.573759733957013) q[0];
ry(-1.3638808633188406) q[1];
cx q[0],q[1];
ry(2.7240945648675012) q[0];
ry(-2.7494267770562533) q[1];
cx q[0],q[1];
ry(-1.8015687281774708) q[1];
ry(-2.1301585642446113) q[2];
cx q[1],q[2];
ry(0.4154781862534032) q[1];
ry(0.19692763604215902) q[2];
cx q[1],q[2];
ry(0.3701942532706983) q[2];
ry(3.0885140189072255) q[3];
cx q[2],q[3];
ry(-0.35049701954710777) q[2];
ry(2.5835807777471387) q[3];
cx q[2],q[3];
ry(-1.2219135668092838) q[3];
ry(0.3905528047566549) q[4];
cx q[3],q[4];
ry(-3.0109240256915952) q[3];
ry(-0.1864742155562813) q[4];
cx q[3],q[4];
ry(1.424568441103187) q[4];
ry(-0.2019301824818065) q[5];
cx q[4],q[5];
ry(-0.06677746261963244) q[4];
ry(0.5558084550724364) q[5];
cx q[4],q[5];
ry(0.2521804162968086) q[5];
ry(0.41833426220207875) q[6];
cx q[5],q[6];
ry(-1.1840737517861963) q[5];
ry(-3.1250406247077036) q[6];
cx q[5],q[6];
ry(0.1474275216339391) q[6];
ry(-0.6869768101644615) q[7];
cx q[6],q[7];
ry(-0.29469520190103005) q[6];
ry(-1.289642772393555) q[7];
cx q[6],q[7];
ry(2.995960216549416) q[7];
ry(-1.3075018224821315) q[8];
cx q[7],q[8];
ry(3.034243112428472) q[7];
ry(0.16835874353699232) q[8];
cx q[7],q[8];
ry(-0.846027076630698) q[8];
ry(-2.5391823093381323) q[9];
cx q[8],q[9];
ry(-0.6636208448663901) q[8];
ry(1.335624964672711) q[9];
cx q[8],q[9];
ry(2.039230194249489) q[9];
ry(-1.5017369946440635) q[10];
cx q[9],q[10];
ry(0.9596666962646531) q[9];
ry(-0.3544172530023157) q[10];
cx q[9],q[10];
ry(-1.8821338985950922) q[10];
ry(-1.2585537567535363) q[11];
cx q[10],q[11];
ry(-0.6188388824518416) q[10];
ry(-1.524974923959858) q[11];
cx q[10],q[11];
ry(1.056453540724867) q[0];
ry(1.079312390479262) q[1];
cx q[0],q[1];
ry(1.7519749656388486) q[0];
ry(1.5738156347448669) q[1];
cx q[0],q[1];
ry(-1.295260169419999) q[1];
ry(-1.2725167961142307) q[2];
cx q[1],q[2];
ry(0.455247305324499) q[1];
ry(-1.8707907268146269) q[2];
cx q[1],q[2];
ry(2.634404033555962) q[2];
ry(1.1283532751562815) q[3];
cx q[2],q[3];
ry(-0.22565789277886095) q[2];
ry(-2.6369723995345713) q[3];
cx q[2],q[3];
ry(2.575217570314205) q[3];
ry(-1.9208133274998413) q[4];
cx q[3],q[4];
ry(-1.502494999169638) q[3];
ry(-0.0024023346809712016) q[4];
cx q[3],q[4];
ry(-1.420933069752461) q[4];
ry(-0.2322577164725241) q[5];
cx q[4],q[5];
ry(0.019901246885240215) q[4];
ry(1.5871408367858866) q[5];
cx q[4],q[5];
ry(-3.0913886406400954) q[5];
ry(1.1156436238034964) q[6];
cx q[5],q[6];
ry(-1.6121829074130052) q[5];
ry(2.9536513657598555) q[6];
cx q[5],q[6];
ry(-0.4193832658822802) q[6];
ry(0.5882896246841538) q[7];
cx q[6],q[7];
ry(-0.8027245502923356) q[6];
ry(-0.9659422128609247) q[7];
cx q[6],q[7];
ry(3.1134865823234446) q[7];
ry(2.0321858775588737) q[8];
cx q[7],q[8];
ry(0.49282247654199024) q[7];
ry(2.998281504455834) q[8];
cx q[7],q[8];
ry(-1.094170273364198) q[8];
ry(-1.152866315010324) q[9];
cx q[8],q[9];
ry(-2.347433230068848) q[8];
ry(0.8625324339423617) q[9];
cx q[8],q[9];
ry(2.711169149375802) q[9];
ry(1.0408769496709218) q[10];
cx q[9],q[10];
ry(1.3480056653269246) q[9];
ry(1.4470098966698182) q[10];
cx q[9],q[10];
ry(0.17759614973827453) q[10];
ry(-0.7839970483938068) q[11];
cx q[10],q[11];
ry(2.221897844511932) q[10];
ry(2.418872028900676) q[11];
cx q[10],q[11];
ry(-1.2007530948463705) q[0];
ry(1.737655933713511) q[1];
cx q[0],q[1];
ry(-1.7046605731480762) q[0];
ry(-2.8415727022586874) q[1];
cx q[0],q[1];
ry(0.7329810734616942) q[1];
ry(1.3239232726961614) q[2];
cx q[1],q[2];
ry(0.767764495198074) q[1];
ry(-0.0579763836912871) q[2];
cx q[1],q[2];
ry(-1.1006816113255242) q[2];
ry(1.5879572894367973) q[3];
cx q[2],q[3];
ry(1.1061778560241828) q[2];
ry(-0.8636673915386313) q[3];
cx q[2],q[3];
ry(-0.10787677942979013) q[3];
ry(1.6370921218792054) q[4];
cx q[3],q[4];
ry(-3.01482794869257) q[3];
ry(-2.8553077125301307) q[4];
cx q[3],q[4];
ry(-1.7087814088914381) q[4];
ry(2.068080311676968) q[5];
cx q[4],q[5];
ry(0.10162945812076796) q[4];
ry(2.939958126000947) q[5];
cx q[4],q[5];
ry(1.338465810830388) q[5];
ry(-1.1813275644523085) q[6];
cx q[5],q[6];
ry(0.5582575460120509) q[5];
ry(2.8361007239646217) q[6];
cx q[5],q[6];
ry(-1.3275652906095425) q[6];
ry(2.932598186405189) q[7];
cx q[6],q[7];
ry(-2.5021508835676074) q[6];
ry(0.34501084205745963) q[7];
cx q[6],q[7];
ry(0.03700015147930598) q[7];
ry(2.490619703767166) q[8];
cx q[7],q[8];
ry(-2.581213121176753) q[7];
ry(-0.07171511513201878) q[8];
cx q[7],q[8];
ry(2.3232056089265605) q[8];
ry(1.6118905165765713) q[9];
cx q[8],q[9];
ry(-0.2326486421271845) q[8];
ry(-1.9740103780498475) q[9];
cx q[8],q[9];
ry(-0.4173021664900993) q[9];
ry(-0.2876084630894327) q[10];
cx q[9],q[10];
ry(-1.5796996102599108) q[9];
ry(-2.9637236161905722) q[10];
cx q[9],q[10];
ry(-2.9104773059932567) q[10];
ry(-0.8907551799449961) q[11];
cx q[10],q[11];
ry(-2.059069786607693) q[10];
ry(-2.9015845849273596) q[11];
cx q[10],q[11];
ry(-2.933635696169233) q[0];
ry(2.0170337060273322) q[1];
cx q[0],q[1];
ry(-0.8867865584495086) q[0];
ry(-2.9515534801167322) q[1];
cx q[0],q[1];
ry(2.5478069549594027) q[1];
ry(-1.5763722016113149) q[2];
cx q[1],q[2];
ry(-1.7069519337004524) q[1];
ry(2.6225271807036954) q[2];
cx q[1],q[2];
ry(-2.990552915790327) q[2];
ry(1.229065232257522) q[3];
cx q[2],q[3];
ry(-0.20682192530813204) q[2];
ry(-3.0658591897458507) q[3];
cx q[2],q[3];
ry(1.670868525829066) q[3];
ry(-1.4027104562204702) q[4];
cx q[3],q[4];
ry(2.920283125350434) q[3];
ry(0.7636411295151166) q[4];
cx q[3],q[4];
ry(0.0874432911419653) q[4];
ry(2.621524318095671) q[5];
cx q[4],q[5];
ry(-0.2468164164532274) q[4];
ry(2.717154864268399) q[5];
cx q[4],q[5];
ry(3.121273477528257) q[5];
ry(0.03403239355736721) q[6];
cx q[5],q[6];
ry(-3.080636344140347) q[5];
ry(-0.3154734350860331) q[6];
cx q[5],q[6];
ry(0.5406936360466288) q[6];
ry(-1.8655503234734407) q[7];
cx q[6],q[7];
ry(0.5016761845237205) q[6];
ry(1.8252543158468528) q[7];
cx q[6],q[7];
ry(0.9115502708540777) q[7];
ry(1.0880515742418848) q[8];
cx q[7],q[8];
ry(3.0546608492913845) q[7];
ry(-3.0309393365629584) q[8];
cx q[7],q[8];
ry(-2.9455610288310266) q[8];
ry(-1.920061141386412) q[9];
cx q[8],q[9];
ry(-0.309794049090395) q[8];
ry(0.8937002197513146) q[9];
cx q[8],q[9];
ry(-2.646702712073657) q[9];
ry(2.518094630243752) q[10];
cx q[9],q[10];
ry(0.5417881075519999) q[9];
ry(-1.2622143036567781) q[10];
cx q[9],q[10];
ry(-1.0812076952520033) q[10];
ry(0.4460160493720835) q[11];
cx q[10],q[11];
ry(2.8022841694822915) q[10];
ry(-1.4063706867599546) q[11];
cx q[10],q[11];
ry(2.0760282983393687) q[0];
ry(-1.0874102827554681) q[1];
cx q[0],q[1];
ry(-2.4151960747928762) q[0];
ry(2.830930451306261) q[1];
cx q[0],q[1];
ry(0.9853801311969274) q[1];
ry(2.1529359529131282) q[2];
cx q[1],q[2];
ry(2.9071372255394885) q[1];
ry(0.4914977564852698) q[2];
cx q[1],q[2];
ry(-2.5926620423007405) q[2];
ry(1.0929204562675539) q[3];
cx q[2],q[3];
ry(1.2754145716814769) q[2];
ry(1.5475202996788822) q[3];
cx q[2],q[3];
ry(0.00450609170185956) q[3];
ry(1.6412302785430217) q[4];
cx q[3],q[4];
ry(3.0937620000944532) q[3];
ry(3.117750530116298) q[4];
cx q[3],q[4];
ry(1.2939138358281095) q[4];
ry(2.3586351715494156) q[5];
cx q[4],q[5];
ry(1.199891549343281) q[4];
ry(-2.4801795861935942) q[5];
cx q[4],q[5];
ry(2.1392274700685148) q[5];
ry(-0.7701694271871297) q[6];
cx q[5],q[6];
ry(-0.944279864534457) q[5];
ry(-0.17316254029962094) q[6];
cx q[5],q[6];
ry(1.4040729154043063) q[6];
ry(1.154739350861286) q[7];
cx q[6],q[7];
ry(-2.7712720485841267) q[6];
ry(-1.5124906421157913) q[7];
cx q[6],q[7];
ry(0.6944500105680383) q[7];
ry(2.35144984863745) q[8];
cx q[7],q[8];
ry(-0.332746923147929) q[7];
ry(-2.6710361192792864) q[8];
cx q[7],q[8];
ry(-1.824529421603856) q[8];
ry(2.90464894706481) q[9];
cx q[8],q[9];
ry(0.46085614815790976) q[8];
ry(2.0443268944726647) q[9];
cx q[8],q[9];
ry(-2.345623108432002) q[9];
ry(1.5380248695985477) q[10];
cx q[9],q[10];
ry(2.001808240811231) q[9];
ry(-1.5520829198016983) q[10];
cx q[9],q[10];
ry(-1.5553162340730744) q[10];
ry(0.7180728990506652) q[11];
cx q[10],q[11];
ry(-3.1363952036292546) q[10];
ry(-0.6569129336363371) q[11];
cx q[10],q[11];
ry(0.9999389089840879) q[0];
ry(-2.303456932733721) q[1];
cx q[0],q[1];
ry(0.49089725735526346) q[0];
ry(0.4071714047926945) q[1];
cx q[0],q[1];
ry(-0.4983219621471297) q[1];
ry(-2.944800141615341) q[2];
cx q[1],q[2];
ry(0.582937195523501) q[1];
ry(0.857580731366148) q[2];
cx q[1],q[2];
ry(-1.6929663051954107) q[2];
ry(2.262943373927066) q[3];
cx q[2],q[3];
ry(3.1100959662168726) q[2];
ry(1.4359102618851556) q[3];
cx q[2],q[3];
ry(-2.7595414715074984) q[3];
ry(-2.049039626766699) q[4];
cx q[3],q[4];
ry(3.129377857136655) q[3];
ry(0.03724921867457187) q[4];
cx q[3],q[4];
ry(-1.4322043211853765) q[4];
ry(-1.0364411136614775) q[5];
cx q[4],q[5];
ry(-0.051418250486511496) q[4];
ry(0.8996174631257503) q[5];
cx q[4],q[5];
ry(-1.0498736724183793) q[5];
ry(-0.9921114190011828) q[6];
cx q[5],q[6];
ry(-2.718828179558161) q[5];
ry(-0.1543668711244451) q[6];
cx q[5],q[6];
ry(2.2468216330313084) q[6];
ry(-2.9657734692299313) q[7];
cx q[6],q[7];
ry(0.06702586870791706) q[6];
ry(0.6194970431284119) q[7];
cx q[6],q[7];
ry(-0.1892169804956252) q[7];
ry(2.149312673338937) q[8];
cx q[7],q[8];
ry(-3.0742469682315696) q[7];
ry(1.9398622984018241) q[8];
cx q[7],q[8];
ry(-0.5390606851923785) q[8];
ry(3.0480211886141544) q[9];
cx q[8],q[9];
ry(2.1685200600155947) q[8];
ry(-1.7612285607086138) q[9];
cx q[8],q[9];
ry(1.0673816214876473) q[9];
ry(2.9598037154139596) q[10];
cx q[9],q[10];
ry(1.1966918975312737) q[9];
ry(2.29820699582581) q[10];
cx q[9],q[10];
ry(-1.5121653345437986) q[10];
ry(-0.31994666515561576) q[11];
cx q[10],q[11];
ry(-2.5608797218061348) q[10];
ry(-0.2649653087480459) q[11];
cx q[10],q[11];
ry(0.798145367265405) q[0];
ry(3.0611855065903586) q[1];
cx q[0],q[1];
ry(1.4893080198513364) q[0];
ry(1.8951887551876339) q[1];
cx q[0],q[1];
ry(2.125305494198801) q[1];
ry(2.9898284450664083) q[2];
cx q[1],q[2];
ry(0.8644853208523373) q[1];
ry(1.91573730395719) q[2];
cx q[1],q[2];
ry(1.870374646284305) q[2];
ry(-0.4608743001066564) q[3];
cx q[2],q[3];
ry(-2.2524239783960827) q[2];
ry(-2.660058335321987) q[3];
cx q[2],q[3];
ry(1.374841921952199) q[3];
ry(1.9630381060940905) q[4];
cx q[3],q[4];
ry(1.1262388100026277) q[3];
ry(-3.0891917287602566) q[4];
cx q[3],q[4];
ry(0.9016053951194088) q[4];
ry(1.8530625941464312) q[5];
cx q[4],q[5];
ry(-0.005909931718833583) q[4];
ry(-0.04548287748853817) q[5];
cx q[4],q[5];
ry(2.0556367753194857) q[5];
ry(-1.5869819006458243) q[6];
cx q[5],q[6];
ry(1.1167410938558535) q[5];
ry(-2.003573429855826) q[6];
cx q[5],q[6];
ry(-3.1312813256795677) q[6];
ry(-1.7501680339866104) q[7];
cx q[6],q[7];
ry(0.5852768483981239) q[6];
ry(0.024293514857729015) q[7];
cx q[6],q[7];
ry(1.778887742019883) q[7];
ry(-1.4077269678292277) q[8];
cx q[7],q[8];
ry(2.13784845830492) q[7];
ry(-0.6052898282086406) q[8];
cx q[7],q[8];
ry(-1.2862010226786722) q[8];
ry(-0.8513680175732898) q[9];
cx q[8],q[9];
ry(-0.5466945742935714) q[8];
ry(3.048612219081543) q[9];
cx q[8],q[9];
ry(2.321739760668678) q[9];
ry(-3.087195656376597) q[10];
cx q[9],q[10];
ry(-1.37083184091306) q[9];
ry(2.7069109101101003) q[10];
cx q[9],q[10];
ry(-1.445571322253744) q[10];
ry(1.0527299253055162) q[11];
cx q[10],q[11];
ry(-1.8981140688933982) q[10];
ry(2.454432790567325) q[11];
cx q[10],q[11];
ry(-1.1800315640350165) q[0];
ry(0.7464823453887903) q[1];
cx q[0],q[1];
ry(-0.8432074080948188) q[0];
ry(-2.9223874355397648) q[1];
cx q[0],q[1];
ry(1.3026022357080196) q[1];
ry(2.0101976752609976) q[2];
cx q[1],q[2];
ry(-0.6813226551199841) q[1];
ry(-1.354923615734664) q[2];
cx q[1],q[2];
ry(-2.7494143822986046) q[2];
ry(1.4498774985112135) q[3];
cx q[2],q[3];
ry(-3.1036041820122495) q[2];
ry(0.5009514379650718) q[3];
cx q[2],q[3];
ry(1.6892808234087004) q[3];
ry(-0.9046459576752767) q[4];
cx q[3],q[4];
ry(-1.994550963988116) q[3];
ry(-3.0416330892772425) q[4];
cx q[3],q[4];
ry(0.7645951826233528) q[4];
ry(-0.5992991449224184) q[5];
cx q[4],q[5];
ry(0.8782056058106172) q[4];
ry(-2.260774794830743) q[5];
cx q[4],q[5];
ry(-0.7029007789459133) q[5];
ry(-0.8423754057345906) q[6];
cx q[5],q[6];
ry(3.080929611048763) q[5];
ry(-2.7855520046681534) q[6];
cx q[5],q[6];
ry(-0.9759123016419303) q[6];
ry(2.0837509456753796) q[7];
cx q[6],q[7];
ry(3.10928445645911) q[6];
ry(-0.02998262703953003) q[7];
cx q[6],q[7];
ry(1.8410668316846115) q[7];
ry(-2.623636530937525) q[8];
cx q[7],q[8];
ry(-0.5244038490610281) q[7];
ry(0.43945111087593247) q[8];
cx q[7],q[8];
ry(-0.6442990621921769) q[8];
ry(-2.4545216138436237) q[9];
cx q[8],q[9];
ry(-0.9688909107874952) q[8];
ry(0.6433114679085509) q[9];
cx q[8],q[9];
ry(-0.31773908618179786) q[9];
ry(-1.535035906271326) q[10];
cx q[9],q[10];
ry(-2.641210032633763) q[9];
ry(-2.724181824663446) q[10];
cx q[9],q[10];
ry(2.517029616783759) q[10];
ry(-0.38063022084140125) q[11];
cx q[10],q[11];
ry(-1.5700488515179056) q[10];
ry(-2.611342223664175) q[11];
cx q[10],q[11];
ry(-1.577516484257636) q[0];
ry(1.7591462826678832) q[1];
cx q[0],q[1];
ry(-0.7251895407372082) q[0];
ry(-1.5569450491228904) q[1];
cx q[0],q[1];
ry(-1.4741477284503774) q[1];
ry(0.9809958013710336) q[2];
cx q[1],q[2];
ry(2.293825042651947) q[1];
ry(0.43177029912713655) q[2];
cx q[1],q[2];
ry(0.14449612183608096) q[2];
ry(-1.4592922695556556) q[3];
cx q[2],q[3];
ry(-2.828514664146847) q[2];
ry(0.9868190140435243) q[3];
cx q[2],q[3];
ry(-2.0175690679797187) q[3];
ry(0.8454477400410773) q[4];
cx q[3],q[4];
ry(-0.014006761051171601) q[3];
ry(-0.07153071367236485) q[4];
cx q[3],q[4];
ry(1.4833704516509103) q[4];
ry(-2.9685774251905777) q[5];
cx q[4],q[5];
ry(0.19371708289557088) q[4];
ry(-3.05152144960499) q[5];
cx q[4],q[5];
ry(-2.547113171359048) q[5];
ry(0.17994650928745287) q[6];
cx q[5],q[6];
ry(-2.164162421235506) q[5];
ry(1.3965795736025255) q[6];
cx q[5],q[6];
ry(1.7341590897837957) q[6];
ry(-1.6101264593308662) q[7];
cx q[6],q[7];
ry(2.712969334042562) q[6];
ry(0.050943696020127084) q[7];
cx q[6],q[7];
ry(-0.818249831940887) q[7];
ry(-0.46477763046651627) q[8];
cx q[7],q[8];
ry(0.16730378594275308) q[7];
ry(3.100981803898994) q[8];
cx q[7],q[8];
ry(0.44452619754092737) q[8];
ry(-1.554773395795685) q[9];
cx q[8],q[9];
ry(-0.8926637620166424) q[8];
ry(-2.1068139274778486) q[9];
cx q[8],q[9];
ry(1.2072010810952185) q[9];
ry(1.7268004883908379) q[10];
cx q[9],q[10];
ry(1.3450595401377907) q[9];
ry(-1.6308648064521953) q[10];
cx q[9],q[10];
ry(-1.6819904953166382) q[10];
ry(0.3834686648466354) q[11];
cx q[10],q[11];
ry(2.570455939758459) q[10];
ry(-2.1995569309536407) q[11];
cx q[10],q[11];
ry(-0.2501247786717832) q[0];
ry(2.687892130831609) q[1];
cx q[0],q[1];
ry(-2.119666934513231) q[0];
ry(-2.7539482318648485) q[1];
cx q[0],q[1];
ry(2.7226409075919267) q[1];
ry(-0.922022136951088) q[2];
cx q[1],q[2];
ry(-0.7829187143522248) q[1];
ry(2.7499417158485358) q[2];
cx q[1],q[2];
ry(-3.035711341470124) q[2];
ry(0.4613154914399846) q[3];
cx q[2],q[3];
ry(-2.832289656276438) q[2];
ry(-2.964099266433488) q[3];
cx q[2],q[3];
ry(1.9432800570646072) q[3];
ry(-2.4463152118054396) q[4];
cx q[3],q[4];
ry(-3.1266417437113136) q[3];
ry(3.0197703078890155) q[4];
cx q[3],q[4];
ry(-0.6085695136239314) q[4];
ry(-2.1977406318431543) q[5];
cx q[4],q[5];
ry(1.063913092086586) q[4];
ry(-2.266696974490609) q[5];
cx q[4],q[5];
ry(-2.4378986934688855) q[5];
ry(1.2373446509394965) q[6];
cx q[5],q[6];
ry(-0.19939886208615665) q[5];
ry(-0.7269665365117904) q[6];
cx q[5],q[6];
ry(-2.5510707063910645) q[6];
ry(1.5558406707469035) q[7];
cx q[6],q[7];
ry(2.8847463208978312) q[6];
ry(-0.22102986844798433) q[7];
cx q[6],q[7];
ry(2.4881817351088027) q[7];
ry(-2.2297367349744093) q[8];
cx q[7],q[8];
ry(-0.09049889700469796) q[7];
ry(-0.06812663729526917) q[8];
cx q[7],q[8];
ry(-2.292394646360046) q[8];
ry(0.4117061246287291) q[9];
cx q[8],q[9];
ry(1.5023361021584727) q[8];
ry(-0.7054303257370813) q[9];
cx q[8],q[9];
ry(-2.9485083183244933) q[9];
ry(-2.2472241176233445) q[10];
cx q[9],q[10];
ry(1.8671878310017986) q[9];
ry(1.8773197749141293) q[10];
cx q[9],q[10];
ry(0.9688787543332547) q[10];
ry(0.8456114540518831) q[11];
cx q[10],q[11];
ry(2.0248906296621945) q[10];
ry(0.8124337955414839) q[11];
cx q[10],q[11];
ry(1.1083545532747987) q[0];
ry(-3.099318009472126) q[1];
cx q[0],q[1];
ry(2.6930109988958164) q[0];
ry(3.1007983569695834) q[1];
cx q[0],q[1];
ry(1.720483692507204) q[1];
ry(1.1245843452790636) q[2];
cx q[1],q[2];
ry(0.2752713278640284) q[1];
ry(-0.3295437735297435) q[2];
cx q[1],q[2];
ry(3.1064949364841996) q[2];
ry(-1.1027350513429173) q[3];
cx q[2],q[3];
ry(-2.0586516551941347) q[2];
ry(0.37886753622736047) q[3];
cx q[2],q[3];
ry(0.19355970452304921) q[3];
ry(1.229215089963882) q[4];
cx q[3],q[4];
ry(3.009210550579039) q[3];
ry(-2.9835724847775538) q[4];
cx q[3],q[4];
ry(-0.6530887400267059) q[4];
ry(-1.3264957931703536) q[5];
cx q[4],q[5];
ry(2.9730130240679475) q[4];
ry(-1.3628488409879287) q[5];
cx q[4],q[5];
ry(1.944331492677006) q[5];
ry(-1.4675085482763657) q[6];
cx q[5],q[6];
ry(-0.09196925676816027) q[5];
ry(-0.08121136752509628) q[6];
cx q[5],q[6];
ry(-1.2160678868334065) q[6];
ry(-3.0045463808874953) q[7];
cx q[6],q[7];
ry(0.5315550663434792) q[6];
ry(1.3602177761969239) q[7];
cx q[6],q[7];
ry(-2.4709207378580693) q[7];
ry(3.1040370529867665) q[8];
cx q[7],q[8];
ry(1.0395223915113192) q[7];
ry(0.08800339795140227) q[8];
cx q[7],q[8];
ry(2.4798194075003934) q[8];
ry(2.7069320924490756) q[9];
cx q[8],q[9];
ry(1.8687178239351034) q[8];
ry(1.706588735739764) q[9];
cx q[8],q[9];
ry(-1.473496471958573) q[9];
ry(-2.2594723380512853) q[10];
cx q[9],q[10];
ry(1.9829287212679472) q[9];
ry(-1.076295836457728) q[10];
cx q[9],q[10];
ry(-1.795642334005864) q[10];
ry(-1.5230003617543337) q[11];
cx q[10],q[11];
ry(-2.2549850023575626) q[10];
ry(0.8820326931082647) q[11];
cx q[10],q[11];
ry(-0.46661757534364956) q[0];
ry(-0.4957142269352363) q[1];
cx q[0],q[1];
ry(1.067319482750289) q[0];
ry(2.6796419923668515) q[1];
cx q[0],q[1];
ry(-1.9888011632436429) q[1];
ry(-2.951407383298747) q[2];
cx q[1],q[2];
ry(1.5485626587483656) q[1];
ry(-0.48994955558048864) q[2];
cx q[1],q[2];
ry(-0.12163465928365547) q[2];
ry(1.1532334462805576) q[3];
cx q[2],q[3];
ry(-0.4825729175478206) q[2];
ry(-1.0325177086868655) q[3];
cx q[2],q[3];
ry(0.9286256650922375) q[3];
ry(2.132723858118177) q[4];
cx q[3],q[4];
ry(2.9909377313616634) q[3];
ry(-0.524092835991386) q[4];
cx q[3],q[4];
ry(1.199850065600471) q[4];
ry(0.3083561361644659) q[5];
cx q[4],q[5];
ry(-2.9931479035106925) q[4];
ry(-1.4437657798059862) q[5];
cx q[4],q[5];
ry(0.15525242186548005) q[5];
ry(2.80358515629198) q[6];
cx q[5],q[6];
ry(2.4981308331125716) q[5];
ry(-0.3753647481414353) q[6];
cx q[5],q[6];
ry(2.977785875129989) q[6];
ry(-0.5491405446360967) q[7];
cx q[6],q[7];
ry(0.1416054668846561) q[6];
ry(3.0978692883964265) q[7];
cx q[6],q[7];
ry(2.5242508152608925) q[7];
ry(0.1515901480783315) q[8];
cx q[7],q[8];
ry(-3.097513119344994) q[7];
ry(-1.2644894926233081) q[8];
cx q[7],q[8];
ry(2.619725007581859) q[8];
ry(1.6466213123455706) q[9];
cx q[8],q[9];
ry(0.6684858460895821) q[8];
ry(-3.0779031481985677) q[9];
cx q[8],q[9];
ry(-0.30445810463477907) q[9];
ry(0.605456265367133) q[10];
cx q[9],q[10];
ry(-1.2582768092882965) q[9];
ry(-1.766742278770634) q[10];
cx q[9],q[10];
ry(2.6088140569776623) q[10];
ry(3.1317333626495363) q[11];
cx q[10],q[11];
ry(-2.914552126464663) q[10];
ry(-2.793804211912732) q[11];
cx q[10],q[11];
ry(-2.7739055373409633) q[0];
ry(-1.817934953817059) q[1];
cx q[0],q[1];
ry(3.1160975727507947) q[0];
ry(-0.008971847154823043) q[1];
cx q[0],q[1];
ry(0.8903086654134078) q[1];
ry(2.706941176074808) q[2];
cx q[1],q[2];
ry(2.4337512034970077) q[1];
ry(-2.7505237295130627) q[2];
cx q[1],q[2];
ry(0.49867209566036536) q[2];
ry(0.8989818450834958) q[3];
cx q[2],q[3];
ry(2.7628257337622766) q[2];
ry(-2.848658175829953) q[3];
cx q[2],q[3];
ry(0.76164543668063) q[3];
ry(-2.7641119267412564) q[4];
cx q[3],q[4];
ry(0.6387002299340976) q[3];
ry(-1.2436730312581021) q[4];
cx q[3],q[4];
ry(1.7255758496270555) q[4];
ry(0.15980649947112635) q[5];
cx q[4],q[5];
ry(3.138992412686459) q[4];
ry(-3.08073347076935) q[5];
cx q[4],q[5];
ry(-1.849896445475827) q[5];
ry(2.8013024875047656) q[6];
cx q[5],q[6];
ry(-0.943425735221485) q[5];
ry(3.098482780269331) q[6];
cx q[5],q[6];
ry(2.581859233756632) q[6];
ry(-0.000512459548095008) q[7];
cx q[6],q[7];
ry(-0.0854338160382153) q[6];
ry(-3.106576826383004) q[7];
cx q[6],q[7];
ry(2.4242563875538603) q[7];
ry(2.56149688750684) q[8];
cx q[7],q[8];
ry(1.0758469021461747) q[7];
ry(0.3409575122668027) q[8];
cx q[7],q[8];
ry(-1.0142835087841433) q[8];
ry(-0.6297743946295933) q[9];
cx q[8],q[9];
ry(2.430620440931552) q[8];
ry(0.05655585099416274) q[9];
cx q[8],q[9];
ry(0.209650959219689) q[9];
ry(-2.8410332053035194) q[10];
cx q[9],q[10];
ry(-0.8908655672877711) q[9];
ry(2.824720611333059) q[10];
cx q[9],q[10];
ry(1.689495468005008) q[10];
ry(-0.2401675447220917) q[11];
cx q[10],q[11];
ry(-2.199372904610724) q[10];
ry(-1.3191489356353865) q[11];
cx q[10],q[11];
ry(-2.4084497125537307) q[0];
ry(-0.9940014473035742) q[1];
cx q[0],q[1];
ry(0.2367346649036639) q[0];
ry(1.9533724812547595) q[1];
cx q[0],q[1];
ry(-2.1676667700358783) q[1];
ry(-1.5117863505069558) q[2];
cx q[1],q[2];
ry(2.154800540047585) q[1];
ry(2.4533051212469648) q[2];
cx q[1],q[2];
ry(1.4362524715494889) q[2];
ry(-1.52262068440168) q[3];
cx q[2],q[3];
ry(-1.998160018908088) q[2];
ry(-2.938554345610276) q[3];
cx q[2],q[3];
ry(-2.400627993775837) q[3];
ry(-1.368213117842501) q[4];
cx q[3],q[4];
ry(-1.1299547676772228) q[3];
ry(1.159436266550711) q[4];
cx q[3],q[4];
ry(3.0541735334661957) q[4];
ry(0.6335789587192027) q[5];
cx q[4],q[5];
ry(3.1362858349748857) q[4];
ry(-0.19494938429088812) q[5];
cx q[4],q[5];
ry(-0.393438962548405) q[5];
ry(-1.4922997854468028) q[6];
cx q[5],q[6];
ry(-3.125967113435793) q[5];
ry(-1.049359821154705) q[6];
cx q[5],q[6];
ry(1.3710236015053994) q[6];
ry(-0.9634252618950788) q[7];
cx q[6],q[7];
ry(-3.0966149948835833) q[6];
ry(-0.027133233408239893) q[7];
cx q[6],q[7];
ry(1.5583716799194542) q[7];
ry(-3.003235711343616) q[8];
cx q[7],q[8];
ry(-1.4942540107309834) q[7];
ry(-1.510796332355139) q[8];
cx q[7],q[8];
ry(0.6119248313875323) q[8];
ry(0.27428685384677776) q[9];
cx q[8],q[9];
ry(-1.924764471161323) q[8];
ry(1.0730221264457267) q[9];
cx q[8],q[9];
ry(-0.2154525471283632) q[9];
ry(-1.5076600030164755) q[10];
cx q[9],q[10];
ry(-0.21173603043344347) q[9];
ry(-2.0751178807343216) q[10];
cx q[9],q[10];
ry(-1.5442447152005288) q[10];
ry(1.7928504307351232) q[11];
cx q[10],q[11];
ry(2.2182935539455473) q[10];
ry(1.3551259486419736) q[11];
cx q[10],q[11];
ry(0.3863432380053967) q[0];
ry(-1.7662119538112862) q[1];
cx q[0],q[1];
ry(-0.1579144703763804) q[0];
ry(1.94611968845221) q[1];
cx q[0],q[1];
ry(0.3214355212961104) q[1];
ry(-2.3604219154593746) q[2];
cx q[1],q[2];
ry(-2.391526548446796) q[1];
ry(1.1257575327615312) q[2];
cx q[1],q[2];
ry(-2.1946902297277426) q[2];
ry(-1.191399884628209) q[3];
cx q[2],q[3];
ry(0.6183554013854446) q[2];
ry(1.0816396686829295) q[3];
cx q[2],q[3];
ry(-2.07615041763355) q[3];
ry(0.5504786181847121) q[4];
cx q[3],q[4];
ry(0.16317892341179574) q[3];
ry(-3.050374398442735) q[4];
cx q[3],q[4];
ry(2.5983261020949677) q[4];
ry(0.914710070607431) q[5];
cx q[4],q[5];
ry(-0.006557055508659658) q[4];
ry(-3.0555404693027084) q[5];
cx q[4],q[5];
ry(-1.8137886779491972) q[5];
ry(-0.45767279354766804) q[6];
cx q[5],q[6];
ry(2.378321226940984) q[5];
ry(0.18236943609881442) q[6];
cx q[5],q[6];
ry(0.6186828098003875) q[6];
ry(-1.4840850297053794) q[7];
cx q[6],q[7];
ry(1.2808676884369494) q[6];
ry(-3.1367415967371) q[7];
cx q[6],q[7];
ry(-1.5682304296448812) q[7];
ry(-0.2687801527935369) q[8];
cx q[7],q[8];
ry(-2.9644855550285643) q[7];
ry(-0.08918925930952872) q[8];
cx q[7],q[8];
ry(0.6607510529939651) q[8];
ry(-0.356807406881898) q[9];
cx q[8],q[9];
ry(-1.2602550231524592) q[8];
ry(-0.8572572574972508) q[9];
cx q[8],q[9];
ry(-0.7996890956866048) q[9];
ry(1.2337562887053062) q[10];
cx q[9],q[10];
ry(-0.4817945062531414) q[9];
ry(-1.3178520033459176) q[10];
cx q[9],q[10];
ry(1.2645552383191545) q[10];
ry(-0.9866497793413496) q[11];
cx q[10],q[11];
ry(2.2808230958340534) q[10];
ry(2.5790199208608557) q[11];
cx q[10],q[11];
ry(-0.09946828872818524) q[0];
ry(1.3617256999569651) q[1];
cx q[0],q[1];
ry(-1.323594265948336) q[0];
ry(-2.4605115652604868) q[1];
cx q[0],q[1];
ry(-0.8264216932839101) q[1];
ry(1.8304350956497917) q[2];
cx q[1],q[2];
ry(2.88061351115733) q[1];
ry(-3.052974096703667) q[2];
cx q[1],q[2];
ry(-1.1112192048743097) q[2];
ry(2.4946039895549865) q[3];
cx q[2],q[3];
ry(0.2333264751640698) q[2];
ry(-1.0516762555817987) q[3];
cx q[2],q[3];
ry(2.01541803950714) q[3];
ry(-0.8299608848284485) q[4];
cx q[3],q[4];
ry(0.725025639046943) q[3];
ry(-0.9809380015052632) q[4];
cx q[3],q[4];
ry(-0.07716282573037248) q[4];
ry(-2.0071760142592625) q[5];
cx q[4],q[5];
ry(0.003484676972703704) q[4];
ry(1.5623307349963746) q[5];
cx q[4],q[5];
ry(-1.813100422242914) q[5];
ry(-3.1284041056832796) q[6];
cx q[5],q[6];
ry(-1.7628329741200828) q[5];
ry(1.4595490905682422) q[6];
cx q[5],q[6];
ry(1.0363305295498915) q[6];
ry(0.3536736715417409) q[7];
cx q[6],q[7];
ry(-2.7495261166781586) q[6];
ry(-3.1355506937185984) q[7];
cx q[6],q[7];
ry(-0.7934130501882111) q[7];
ry(-2.8925009065644747) q[8];
cx q[7],q[8];
ry(3.0121068913321913) q[7];
ry(-0.039630195816236495) q[8];
cx q[7],q[8];
ry(2.7284953706059514) q[8];
ry(-0.07757287370662524) q[9];
cx q[8],q[9];
ry(2.840661560150472) q[8];
ry(0.9347597062202252) q[9];
cx q[8],q[9];
ry(0.5676748493084123) q[9];
ry(0.9742840578992329) q[10];
cx q[9],q[10];
ry(-1.2687183378467441) q[9];
ry(-1.1330385313205396) q[10];
cx q[9],q[10];
ry(2.538845222636601) q[10];
ry(-1.7036248847668165) q[11];
cx q[10],q[11];
ry(1.1311820623141269) q[10];
ry(0.31168868443727143) q[11];
cx q[10],q[11];
ry(1.3402512165425906) q[0];
ry(1.0607010612454475) q[1];
cx q[0],q[1];
ry(-2.5454441505446277) q[0];
ry(0.9742452036050191) q[1];
cx q[0],q[1];
ry(2.59256048903436) q[1];
ry(0.9550818021453786) q[2];
cx q[1],q[2];
ry(-0.24478621534634293) q[1];
ry(1.8524411575792217) q[2];
cx q[1],q[2];
ry(0.8816481581520523) q[2];
ry(1.6604854882224993) q[3];
cx q[2],q[3];
ry(0.4052672734424254) q[2];
ry(-0.11884920184238876) q[3];
cx q[2],q[3];
ry(3.0162119878826714) q[3];
ry(-1.5563990362134508) q[4];
cx q[3],q[4];
ry(-0.6915174112523892) q[3];
ry(-1.5344116097830793) q[4];
cx q[3],q[4];
ry(-0.008263962949490278) q[4];
ry(-0.6572698440304369) q[5];
cx q[4],q[5];
ry(-0.0010875487189299093) q[4];
ry(0.12853278198191642) q[5];
cx q[4],q[5];
ry(0.6727079188583405) q[5];
ry(2.9490576790367675) q[6];
cx q[5],q[6];
ry(0.8406616392321853) q[5];
ry(-1.6815813500414944) q[6];
cx q[5],q[6];
ry(0.13565396426685614) q[6];
ry(0.03900071689560569) q[7];
cx q[6],q[7];
ry(2.9752307193832266) q[6];
ry(1.5098568078156918) q[7];
cx q[6],q[7];
ry(3.012769857169622) q[7];
ry(-2.6547624797265517) q[8];
cx q[7],q[8];
ry(0.03984516771228641) q[7];
ry(-3.03255771339703) q[8];
cx q[7],q[8];
ry(0.5257631544270381) q[8];
ry(-2.9752571027146493) q[9];
cx q[8],q[9];
ry(0.8539941296236099) q[8];
ry(-1.2342087824983061) q[9];
cx q[8],q[9];
ry(2.863795732398288) q[9];
ry(3.034011210213599) q[10];
cx q[9],q[10];
ry(-2.211492928584607) q[9];
ry(-3.080541967393431) q[10];
cx q[9],q[10];
ry(-1.9274320179172975) q[10];
ry(-2.6985996182202525) q[11];
cx q[10],q[11];
ry(0.4761871358602638) q[10];
ry(1.883353736876793) q[11];
cx q[10],q[11];
ry(-2.7840282623413612) q[0];
ry(2.018405010545749) q[1];
cx q[0],q[1];
ry(-1.78382290927256) q[0];
ry(3.001209402166442) q[1];
cx q[0],q[1];
ry(-1.9054939853033606) q[1];
ry(-1.8763620013176558) q[2];
cx q[1],q[2];
ry(-0.17808899805898282) q[1];
ry(2.252848524615053) q[2];
cx q[1],q[2];
ry(0.4191764007298202) q[2];
ry(1.5804103971680385) q[3];
cx q[2],q[3];
ry(1.196967202135383) q[2];
ry(0.003107903207881177) q[3];
cx q[2],q[3];
ry(2.3537912290598006) q[3];
ry(0.006920411002258645) q[4];
cx q[3],q[4];
ry(-1.0937360741077402) q[3];
ry(-0.017577368346585943) q[4];
cx q[3],q[4];
ry(-2.1681735047463295) q[4];
ry(2.268831584180891) q[5];
cx q[4],q[5];
ry(0.0013030452663480943) q[4];
ry(0.0004490430661819995) q[5];
cx q[4],q[5];
ry(-2.326457651017497) q[5];
ry(-0.20202805369132903) q[6];
cx q[5],q[6];
ry(-1.5318050951396387) q[5];
ry(-3.1395637584036646) q[6];
cx q[5],q[6];
ry(1.7610222275666523) q[6];
ry(-1.4717901720281557) q[7];
cx q[6],q[7];
ry(2.1390107892460475) q[6];
ry(-2.1066836603784367) q[7];
cx q[6],q[7];
ry(-0.14272790307141275) q[7];
ry(0.19799301225314547) q[8];
cx q[7],q[8];
ry(-3.0214935327581696) q[7];
ry(-0.01452457844124958) q[8];
cx q[7],q[8];
ry(2.4716317014762033) q[8];
ry(0.8828307334378476) q[9];
cx q[8],q[9];
ry(-0.8672124605941248) q[8];
ry(1.6980300807027193) q[9];
cx q[8],q[9];
ry(-1.1485860051799839) q[9];
ry(0.012405453609582116) q[10];
cx q[9],q[10];
ry(-1.6779495658366557) q[9];
ry(-1.1430146163871635) q[10];
cx q[9],q[10];
ry(-1.791548195423604) q[10];
ry(1.7126198745071468) q[11];
cx q[10],q[11];
ry(0.7864373058197864) q[10];
ry(1.6211766698606638) q[11];
cx q[10],q[11];
ry(1.1250778002657436) q[0];
ry(-1.0538171029604442) q[1];
cx q[0],q[1];
ry(-1.0976566248395627) q[0];
ry(-1.8288792034264796) q[1];
cx q[0],q[1];
ry(-0.8932556069474415) q[1];
ry(0.7137283854477996) q[2];
cx q[1],q[2];
ry(-1.767404969885046) q[1];
ry(-0.9623451297966584) q[2];
cx q[1],q[2];
ry(-1.3826934423928743) q[2];
ry(-2.354250194368256) q[3];
cx q[2],q[3];
ry(-2.667585998952083) q[2];
ry(-3.1251513728095257) q[3];
cx q[2],q[3];
ry(0.350128161467052) q[3];
ry(-1.6496471608183767) q[4];
cx q[3],q[4];
ry(1.6679528456537955) q[3];
ry(3.052156595936988) q[4];
cx q[3],q[4];
ry(-2.7258426027838465) q[4];
ry(2.7630594275493667) q[5];
cx q[4],q[5];
ry(0.0005552603388164007) q[4];
ry(-3.1328821356917347) q[5];
cx q[4],q[5];
ry(0.8887715775869767) q[5];
ry(-1.8311490762032847) q[6];
cx q[5],q[6];
ry(-2.8527746638030034) q[5];
ry(-3.1368565037091165) q[6];
cx q[5],q[6];
ry(0.11141273997001112) q[6];
ry(-2.980856401729487) q[7];
cx q[6],q[7];
ry(-0.9257227613158441) q[6];
ry(1.8677167140291928) q[7];
cx q[6],q[7];
ry(-1.760727306474329) q[7];
ry(1.5352989162485446) q[8];
cx q[7],q[8];
ry(0.44530987043643766) q[7];
ry(3.1263056765945287) q[8];
cx q[7],q[8];
ry(2.654388992064528) q[8];
ry(-0.28889080327229) q[9];
cx q[8],q[9];
ry(1.3496649323379541) q[8];
ry(-0.25197581548690096) q[9];
cx q[8],q[9];
ry(-2.8223458020973875) q[9];
ry(-1.5812604047804) q[10];
cx q[9],q[10];
ry(1.0319649333397753) q[9];
ry(1.9344447601115562) q[10];
cx q[9],q[10];
ry(-2.792427421576819) q[10];
ry(1.7401569925355398) q[11];
cx q[10],q[11];
ry(1.5841992719182107) q[10];
ry(-0.627499170240985) q[11];
cx q[10],q[11];
ry(-0.48719301540759563) q[0];
ry(-0.8946386510513641) q[1];
cx q[0],q[1];
ry(1.4955424788413412) q[0];
ry(2.722035708894654) q[1];
cx q[0],q[1];
ry(-0.32631046219294024) q[1];
ry(1.7905459153309713) q[2];
cx q[1],q[2];
ry(-1.6724275098291084) q[1];
ry(-0.40179407565805597) q[2];
cx q[1],q[2];
ry(1.438570793351131) q[2];
ry(-1.7009847998378655) q[3];
cx q[2],q[3];
ry(3.139488333978506) q[2];
ry(1.5549747916263328) q[3];
cx q[2],q[3];
ry(2.6583981259439136) q[3];
ry(1.0019629422234455) q[4];
cx q[3],q[4];
ry(1.020886798302299) q[3];
ry(1.9170916909913884) q[4];
cx q[3],q[4];
ry(-2.3836348619159833) q[4];
ry(0.17418949318585764) q[5];
cx q[4],q[5];
ry(-0.007318864130552605) q[4];
ry(-1.6854488992286458) q[5];
cx q[4],q[5];
ry(0.059791572446162476) q[5];
ry(0.340649929254951) q[6];
cx q[5],q[6];
ry(-2.553454583776067) q[5];
ry(-3.0976157431388285) q[6];
cx q[5],q[6];
ry(1.707485907079632) q[6];
ry(-2.570270944438326) q[7];
cx q[6],q[7];
ry(3.0969895058417305) q[6];
ry(2.721964872146064) q[7];
cx q[6],q[7];
ry(-0.4808713820876464) q[7];
ry(0.25459112511810744) q[8];
cx q[7],q[8];
ry(2.507413507577214) q[7];
ry(-3.097818632227377) q[8];
cx q[7],q[8];
ry(-0.7946803693080762) q[8];
ry(0.9956210417777527) q[9];
cx q[8],q[9];
ry(0.34355008019254996) q[8];
ry(1.9095196775324519) q[9];
cx q[8],q[9];
ry(3.016039696262599) q[9];
ry(-0.8412960626964407) q[10];
cx q[9],q[10];
ry(-2.0075606963365544) q[9];
ry(1.8186350877983792) q[10];
cx q[9],q[10];
ry(-1.5326210857537161) q[10];
ry(1.8711724878315978) q[11];
cx q[10],q[11];
ry(-2.3542331536926184) q[10];
ry(-2.228393753738258) q[11];
cx q[10],q[11];
ry(2.553673424124505) q[0];
ry(-1.2476085760949207) q[1];
cx q[0],q[1];
ry(-1.8166862850914167) q[0];
ry(1.7774119612652246) q[1];
cx q[0],q[1];
ry(-1.3776092304307017) q[1];
ry(2.3124153061453074) q[2];
cx q[1],q[2];
ry(1.3608707801952231) q[1];
ry(-1.5713616013779674) q[2];
cx q[1],q[2];
ry(1.6457184345282463) q[2];
ry(2.9894058863131026) q[3];
cx q[2],q[3];
ry(3.1390363082795054) q[2];
ry(0.005677593323073035) q[3];
cx q[2],q[3];
ry(2.970294950360639) q[3];
ry(1.5721878451356792) q[4];
cx q[3],q[4];
ry(0.4840997110937266) q[3];
ry(1.8178726266794771) q[4];
cx q[3],q[4];
ry(1.605070786652984) q[4];
ry(2.7162575192373937) q[5];
cx q[4],q[5];
ry(0.005244246840665119) q[4];
ry(-3.1096444658897853) q[5];
cx q[4],q[5];
ry(-1.8024347013930262) q[5];
ry(3.0341198454983265) q[6];
cx q[5],q[6];
ry(-1.074781306360148) q[5];
ry(1.2046932654966866) q[6];
cx q[5],q[6];
ry(1.3306630261251602) q[6];
ry(2.2500358823047466) q[7];
cx q[6],q[7];
ry(-3.081870385584622) q[6];
ry(0.0027116500635093743) q[7];
cx q[6],q[7];
ry(0.37170917690486216) q[7];
ry(-2.742469157138139) q[8];
cx q[7],q[8];
ry(0.23675710908062278) q[7];
ry(-2.647336715958759) q[8];
cx q[7],q[8];
ry(-2.756650417992103) q[8];
ry(-2.340276317981897) q[9];
cx q[8],q[9];
ry(2.9219323725128503) q[8];
ry(2.3378133129320156) q[9];
cx q[8],q[9];
ry(1.1465010041061605) q[9];
ry(-0.9968037852325979) q[10];
cx q[9],q[10];
ry(2.712205488996756) q[9];
ry(-1.57244223810889) q[10];
cx q[9],q[10];
ry(-2.669892604657334) q[10];
ry(3.0786505643319195) q[11];
cx q[10],q[11];
ry(-1.6374218462504948) q[10];
ry(1.5780572178001213) q[11];
cx q[10],q[11];
ry(-1.5708699755675697) q[0];
ry(-1.523582541236994) q[1];
cx q[0],q[1];
ry(0.011720672254178055) q[0];
ry(-1.5721234702246065) q[1];
cx q[0],q[1];
ry(1.5105823184902227) q[1];
ry(0.6400981395691865) q[2];
cx q[1],q[2];
ry(-2.1854122310955533) q[1];
ry(0.8772714579081065) q[2];
cx q[1],q[2];
ry(-1.998667202747371) q[2];
ry(-1.7078683247357684) q[3];
cx q[2],q[3];
ry(-0.06444466143833658) q[2];
ry(-0.0012259666743894536) q[3];
cx q[2],q[3];
ry(-0.03314436818592448) q[3];
ry(1.5489911823164473) q[4];
cx q[3],q[4];
ry(3.1207334876388564) q[3];
ry(-3.120561828382197) q[4];
cx q[3],q[4];
ry(2.886496236646921) q[4];
ry(-2.67879660224227) q[5];
cx q[4],q[5];
ry(-0.01739312235915576) q[4];
ry(-3.1346807444533216) q[5];
cx q[4],q[5];
ry(-0.5166297473264281) q[5];
ry(2.2735283598247555) q[6];
cx q[5],q[6];
ry(1.0922761761882063) q[5];
ry(-1.2295621525368228) q[6];
cx q[5],q[6];
ry(-0.46426752348242406) q[6];
ry(1.9281382344524252) q[7];
cx q[6],q[7];
ry(3.0600263991929433) q[6];
ry(0.014586675989837872) q[7];
cx q[6],q[7];
ry(2.7027421116507626) q[7];
ry(-1.2814089081167177) q[8];
cx q[7],q[8];
ry(0.6305932985117014) q[7];
ry(2.6727628312282885) q[8];
cx q[7],q[8];
ry(2.9331327890759615) q[8];
ry(1.619057992546587) q[9];
cx q[8],q[9];
ry(-1.58374439211867) q[8];
ry(1.5673090769618279) q[9];
cx q[8],q[9];
ry(-0.0770567009713759) q[9];
ry(0.10105513550024404) q[10];
cx q[9],q[10];
ry(-1.5853548851466792) q[9];
ry(-1.7295539125490544) q[10];
cx q[9],q[10];
ry(2.719762319259595) q[10];
ry(2.942372686911345) q[11];
cx q[10],q[11];
ry(0.01708588164026438) q[10];
ry(1.5648060603650664) q[11];
cx q[10],q[11];
ry(2.031437427499417) q[0];
ry(-2.6978583659302284) q[1];
ry(3.0745035541138237) q[2];
ry(1.8411081900880237) q[3];
ry(-1.3232042707863687) q[4];
ry(2.351050224234155) q[5];
ry(-2.1564857534179476) q[6];
ry(1.9589969475485733) q[7];
ry(-3.005258825963419) q[8];
ry(2.8150593387175697) q[9];
ry(-2.9489489529288178) q[10];
ry(-0.9007757504636739) q[11];