OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.5421242783165932) q[0];
rz(-1.8574578783380638) q[0];
ry(2.5761114431564005) q[1];
rz(2.303175607404297) q[1];
ry(-0.984387962026724) q[2];
rz(1.5291813242124792) q[2];
ry(0.8171858991026512) q[3];
rz(0.8449639808429259) q[3];
ry(0.20759775729105648) q[4];
rz(-2.991528718388611) q[4];
ry(-2.7601719375814127) q[5];
rz(-1.0099350035525787) q[5];
ry(-2.4878244305276387) q[6];
rz(-0.5308064797224112) q[6];
ry(-2.35967034404529) q[7];
rz(1.8193095585504628) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.1868938828792004) q[0];
rz(-0.015164667658441216) q[0];
ry(1.4559602008129033) q[1];
rz(-2.809057238306687) q[1];
ry(-0.646686675103318) q[2];
rz(2.9816746387820694) q[2];
ry(-1.7269407073731085) q[3];
rz(-1.0910462934773306) q[3];
ry(-2.1067333237670844) q[4];
rz(-2.6657194501544046) q[4];
ry(-1.9999377261984623) q[5];
rz(-2.907163367958938) q[5];
ry(-2.7405110605144216) q[6];
rz(-0.11607899740307123) q[6];
ry(2.0891953855324608) q[7];
rz(-0.7436006776449203) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.6895706589904265) q[0];
rz(-2.831064806326203) q[0];
ry(2.2866679706537667) q[1];
rz(0.891248347351941) q[1];
ry(-2.5155596865752425) q[2];
rz(-2.503834659416006) q[2];
ry(0.32959708498814244) q[3];
rz(-1.932939261333602) q[3];
ry(2.286527700392217) q[4];
rz(-1.9625213084950444) q[4];
ry(-0.14224437038589954) q[5];
rz(-2.8235818442024914) q[5];
ry(2.6640486892238107) q[6];
rz(-2.086499362027424) q[6];
ry(0.1981618195356747) q[7];
rz(-0.15154327279406044) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.4679416880741956) q[0];
rz(-1.7538152175648254) q[0];
ry(0.5043941415651911) q[1];
rz(-1.5387291936802114) q[1];
ry(1.7767278703314924) q[2];
rz(-0.4841215257433739) q[2];
ry(-1.2780373175092654) q[3];
rz(3.038853573923077) q[3];
ry(0.934894135546214) q[4];
rz(-2.8013873668386715) q[4];
ry(1.8061854241696773) q[5];
rz(1.8622551345883653) q[5];
ry(-0.7147808206124382) q[6];
rz(1.9861440953835745) q[6];
ry(-2.635187947369886) q[7];
rz(1.2000648118539787) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.7021159063780735) q[0];
rz(3.1179053163461385) q[0];
ry(0.6136524292482474) q[1];
rz(1.4277012738102766) q[1];
ry(-1.4506581271914454) q[2];
rz(-0.7251630914224848) q[2];
ry(-2.010745728893971) q[3];
rz(2.9570003939035567) q[3];
ry(1.7870643836296844) q[4];
rz(-2.345991166233717) q[4];
ry(1.6521298973960592) q[5];
rz(-2.0987828264467936) q[5];
ry(2.50622640519728) q[6];
rz(-0.8321646887636135) q[6];
ry(0.030003067363524316) q[7];
rz(1.6863697581892474) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.1183558726751708) q[0];
rz(-1.976709196751485) q[0];
ry(-1.089368899121653) q[1];
rz(0.8937771144533441) q[1];
ry(-1.9366428610158557) q[2];
rz(0.6637204372712109) q[2];
ry(-0.26364587084918156) q[3];
rz(2.7320196786985806) q[3];
ry(2.2338807665209366) q[4];
rz(-1.557462315752236) q[4];
ry(-1.1037869392515305) q[5];
rz(-2.5532545790123238) q[5];
ry(-0.5091216405568595) q[6];
rz(0.06945606183720088) q[6];
ry(2.717161029051765) q[7];
rz(0.6034763857311186) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.6873574656155825) q[0];
rz(-0.9053811825451309) q[0];
ry(0.40534104344547117) q[1];
rz(0.782598276552304) q[1];
ry(-0.527836029928852) q[2];
rz(0.6905427119476951) q[2];
ry(1.5616756275594286) q[3];
rz(-2.030438081555207) q[3];
ry(1.6923678496049153) q[4];
rz(0.05675370500040024) q[4];
ry(1.3149582768770571) q[5];
rz(-2.5715352686491513) q[5];
ry(-2.2460014536064485) q[6];
rz(-2.2465947862682665) q[6];
ry(-0.4915198622381265) q[7];
rz(0.8397884485562406) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.940655392417248) q[0];
rz(1.6525759859210167) q[0];
ry(2.026734908506108) q[1];
rz(0.6919912401608324) q[1];
ry(-2.7422820338573874) q[2];
rz(2.453321112735074) q[2];
ry(-1.541629141863386) q[3];
rz(-2.4963305210488196) q[3];
ry(-2.4086248156140115) q[4];
rz(0.792413045160421) q[4];
ry(-0.25018037144461225) q[5];
rz(-2.2782999146877323) q[5];
ry(1.301441082166411) q[6];
rz(-2.3136471729712786) q[6];
ry(0.7836579131579141) q[7];
rz(1.5832298563199305) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.4481383162312231) q[0];
rz(-2.355815325838165) q[0];
ry(0.967270632214718) q[1];
rz(2.5271742042095635) q[1];
ry(-1.9395536628550714) q[2];
rz(0.7254684427090643) q[2];
ry(1.5701811882449368) q[3];
rz(1.6766113027564684) q[3];
ry(-2.0919596440546897) q[4];
rz(-0.03965797555120254) q[4];
ry(1.702490744588089) q[5];
rz(2.108967030091312) q[5];
ry(0.2915914642619164) q[6];
rz(-0.39872754845670055) q[6];
ry(0.21501483637211474) q[7];
rz(-2.4516025645795954) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.8028077834499037) q[0];
rz(1.9128156127971303) q[0];
ry(-0.1338952299476414) q[1];
rz(1.534156514701835) q[1];
ry(-1.5519566045968218) q[2];
rz(-2.6860095979248593) q[2];
ry(2.8150607121791533) q[3];
rz(2.342978821016479) q[3];
ry(-2.7938019646867844) q[4];
rz(-1.331027923209694) q[4];
ry(-2.5776269606453033) q[5];
rz(1.1624201040979685) q[5];
ry(1.3076009229575423) q[6];
rz(0.9832407982745581) q[6];
ry(0.04249068096708217) q[7];
rz(-0.8704990966475075) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.7432513514609704) q[0];
rz(-1.5943697565487245) q[0];
ry(-2.5900531191604723) q[1];
rz(-0.06841787621389046) q[1];
ry(-0.10930232056303935) q[2];
rz(-0.3277138042802231) q[2];
ry(3.0523355803147445) q[3];
rz(-1.9416907576413007) q[3];
ry(0.8372726365446643) q[4];
rz(-1.4121948605070382) q[4];
ry(-2.9287010747857276) q[5];
rz(-2.601350318290575) q[5];
ry(1.3692003730788835) q[6];
rz(2.193795074295383) q[6];
ry(-0.1517150848772406) q[7];
rz(0.1774895435150403) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.2760214478347807) q[0];
rz(2.1639103692559205) q[0];
ry(0.78544016454951) q[1];
rz(-1.165669805481569) q[1];
ry(-1.5331129157432386) q[2];
rz(-0.014473757828229456) q[2];
ry(0.6809185601234549) q[3];
rz(2.4200815554379664) q[3];
ry(-0.13764071785759827) q[4];
rz(-2.3448162668350174) q[4];
ry(1.1315959550983132) q[5];
rz(-2.3595168416112067) q[5];
ry(2.972596205998265) q[6];
rz(-0.4283126752229298) q[6];
ry(1.0401828674636326) q[7];
rz(0.38603467833925986) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.211504168528358) q[0];
rz(1.790148189590373) q[0];
ry(-1.1531847918468285) q[1];
rz(2.410699038751797) q[1];
ry(1.3132836073846805) q[2];
rz(-1.6428472774995864) q[2];
ry(-2.0010020075533816) q[3];
rz(-1.4954531834680331) q[3];
ry(-0.08254365720240019) q[4];
rz(2.765585307952254) q[4];
ry(0.2857285111814311) q[5];
rz(2.4049051295370774) q[5];
ry(-2.851089573849179) q[6];
rz(-2.839309309221607) q[6];
ry(2.6040610121042036) q[7];
rz(0.9480401217487788) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.557644659109503) q[0];
rz(-1.6441612752694985) q[0];
ry(-2.244095270899215) q[1];
rz(-2.7055015609496675) q[1];
ry(1.2509630726959255) q[2];
rz(0.7664905383600518) q[2];
ry(0.5466123016241415) q[3];
rz(1.0838493230586668) q[3];
ry(-1.0945238714097165) q[4];
rz(3.099608530546871) q[4];
ry(2.5402392689342106) q[5];
rz(-0.6142269403815322) q[5];
ry(-2.6217214213706233) q[6];
rz(1.079114088085731) q[6];
ry(2.1434194883204993) q[7];
rz(2.1955643198619685) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.2307714007823733) q[0];
rz(-1.2671908195038828) q[0];
ry(3.1083514715596365) q[1];
rz(-0.15788837219528148) q[1];
ry(0.2409901284356551) q[2];
rz(-3.0199310492368774) q[2];
ry(2.3302013711915426) q[3];
rz(2.664186394537467) q[3];
ry(-1.3461403633803821) q[4];
rz(0.20355967424318244) q[4];
ry(-1.6416496491998216) q[5];
rz(-2.5295052624726297) q[5];
ry(2.9552615592911393) q[6];
rz(3.047066440437211) q[6];
ry(1.1394741392292795) q[7];
rz(1.0305268450170626) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.183462703250523) q[0];
rz(2.394247775380856) q[0];
ry(-2.552825128260133) q[1];
rz(-2.6893428952908223) q[1];
ry(2.7858316389681286) q[2];
rz(-0.632594893976398) q[2];
ry(-2.162143491192995) q[3];
rz(-1.7814707477033687) q[3];
ry(-0.8458714625980619) q[4];
rz(2.6921256323977762) q[4];
ry(-0.06591295459252411) q[5];
rz(-0.49149853024908197) q[5];
ry(0.21500939821869294) q[6];
rz(1.0416918508676671) q[6];
ry(-1.30340664862414) q[7];
rz(-1.4368005003077924) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.931470790854864) q[0];
rz(-0.7766053235217889) q[0];
ry(0.6019241120023902) q[1];
rz(-2.8100452143638037) q[1];
ry(1.811154370910149) q[2];
rz(0.9098123917126708) q[2];
ry(-1.3819322699792043) q[3];
rz(-1.3832823427977028) q[3];
ry(0.21208199325233246) q[4];
rz(1.964993218820032) q[4];
ry(1.8828283096982608) q[5];
rz(2.3991952533264387) q[5];
ry(1.7918603494825243) q[6];
rz(-0.17718517887762086) q[6];
ry(-1.7263008425531643) q[7];
rz(-0.13873413268775966) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.26702501982083415) q[0];
rz(-0.9187674020409355) q[0];
ry(-2.9528428625088514) q[1];
rz(-2.889908848237066) q[1];
ry(-2.6278387291779883) q[2];
rz(-2.106509793636471) q[2];
ry(2.8423671089730744) q[3];
rz(1.5339338827451132) q[3];
ry(1.389787209528743) q[4];
rz(3.0049872038460816) q[4];
ry(-2.0383702913481443) q[5];
rz(-0.7163078609903144) q[5];
ry(1.2795266199945141) q[6];
rz(1.8114140293686818) q[6];
ry(-0.48860072369144364) q[7];
rz(-0.0736492638224382) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.9351712183925611) q[0];
rz(-1.7125318840662873) q[0];
ry(0.053200215816517786) q[1];
rz(-2.0516990417214087) q[1];
ry(-0.4105865589705529) q[2];
rz(-2.9572467614758686) q[2];
ry(1.5720959902635514) q[3];
rz(1.2676298581752006) q[3];
ry(-2.368507312221486) q[4];
rz(1.7854410407413417) q[4];
ry(0.3527343876422737) q[5];
rz(2.220134790655367) q[5];
ry(-0.6736316635210934) q[6];
rz(-0.784023831802834) q[6];
ry(2.388411780231658) q[7];
rz(0.1696151346738975) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.6638232325051616) q[0];
rz(1.1179711910600991) q[0];
ry(2.0687477285121014) q[1];
rz(2.1761785885924) q[1];
ry(-1.0708661246689213) q[2];
rz(-1.0175622947398146) q[2];
ry(-2.587272872972763) q[3];
rz(-1.1467471056282375) q[3];
ry(2.3764760200307506) q[4];
rz(-1.9242671718803999) q[4];
ry(-1.2441068176373247) q[5];
rz(0.37709859867734785) q[5];
ry(2.7265619117090956) q[6];
rz(1.1309787683566475) q[6];
ry(1.8368461326392627) q[7];
rz(1.1545633779212439) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.2983559581946222) q[0];
rz(-1.3749718701333276) q[0];
ry(-1.447153060297154) q[1];
rz(-1.5525987326905792) q[1];
ry(-0.1612000771525901) q[2];
rz(0.865982780165452) q[2];
ry(-0.7689208617625216) q[3];
rz(-1.1326710103089115) q[3];
ry(0.9658655091143613) q[4];
rz(-1.2852692448766145) q[4];
ry(-2.679434351493802) q[5];
rz(0.6984734282671665) q[5];
ry(-0.6605849872716254) q[6];
rz(2.040586237979963) q[6];
ry(0.2141053815493569) q[7];
rz(1.2209616767194456) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.6016995884099554) q[0];
rz(2.45553669444376) q[0];
ry(1.664204089876311) q[1];
rz(-0.20742629420922443) q[1];
ry(2.9741895445120923) q[2];
rz(-1.716827358760581) q[2];
ry(-0.6363988845299966) q[3];
rz(-3.1301924528746494) q[3];
ry(-0.3356608185831016) q[4];
rz(-1.8172291196200365) q[4];
ry(1.9290223611837307) q[5];
rz(-0.9913511340495695) q[5];
ry(2.353659734636865) q[6];
rz(0.5332218493763137) q[6];
ry(0.3519939984955897) q[7];
rz(1.9193345653807878) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.38691274245534646) q[0];
rz(0.2659425064466969) q[0];
ry(0.4455029190840137) q[1];
rz(-1.4143630540404644) q[1];
ry(0.9013161867846424) q[2];
rz(1.9270183992263095) q[2];
ry(1.4170858101089827) q[3];
rz(0.0538614733890937) q[3];
ry(0.6837641516852662) q[4];
rz(2.7800491830781615) q[4];
ry(-0.6309622938721521) q[5];
rz(0.815989566412213) q[5];
ry(-1.7783943616540547) q[6];
rz(-0.6409776094738318) q[6];
ry(-1.9871391813001926) q[7];
rz(2.1892568108230477) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.8686048080708115) q[0];
rz(2.9422902383564953) q[0];
ry(-2.9892898539249946) q[1];
rz(0.959822719682468) q[1];
ry(-2.690481889663106) q[2];
rz(-1.2544204286043772) q[2];
ry(2.5359628584132325) q[3];
rz(1.106701529077431) q[3];
ry(0.9267093738063386) q[4];
rz(-0.5104398082683348) q[4];
ry(1.3527485618165442) q[5];
rz(0.6104234490966435) q[5];
ry(-2.4637389108412653) q[6];
rz(1.9213957581836332) q[6];
ry(-2.3426176684311093) q[7];
rz(-1.1669550208878665) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.6122032356803102) q[0];
rz(-0.002850181687285236) q[0];
ry(1.4298595419867135) q[1];
rz(1.6094490895836777) q[1];
ry(1.6015631084793032) q[2];
rz(0.7765714037846421) q[2];
ry(2.5206048039634763) q[3];
rz(-1.071298047235436) q[3];
ry(0.8177886176962708) q[4];
rz(-2.3741350602183844) q[4];
ry(-1.5116542328558886) q[5];
rz(-2.974420135641399) q[5];
ry(-1.9638003231106975) q[6];
rz(0.07788595989951894) q[6];
ry(-2.2485796194212915) q[7];
rz(0.5672550788876229) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.07744596573085055) q[0];
rz(-0.21946960167159849) q[0];
ry(-2.6170139641545935) q[1];
rz(-0.5354961975693024) q[1];
ry(-2.6332899445814935) q[2];
rz(2.8320527015319032) q[2];
ry(2.3300712709657176) q[3];
rz(2.374499408424478) q[3];
ry(-1.5586718715832817) q[4];
rz(-0.7587571392064625) q[4];
ry(-0.7612244440375903) q[5];
rz(0.33433094469334085) q[5];
ry(-2.1190265416892484) q[6];
rz(-1.594863427107753) q[6];
ry(-2.295351866997921) q[7];
rz(-2.0915754632461008) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.2463184616348624) q[0];
rz(-2.552384524228528) q[0];
ry(-3.1401022782415797) q[1];
rz(-0.5167585090503549) q[1];
ry(-1.7816727450212564) q[2];
rz(-0.8931703309478839) q[2];
ry(1.1712472251757813) q[3];
rz(-0.08594843401923492) q[3];
ry(2.4818190037889045) q[4];
rz(0.7127510985886457) q[4];
ry(-2.1352201239117248) q[5];
rz(-2.818006442545071) q[5];
ry(1.7312457022792485) q[6];
rz(1.1324805347433078) q[6];
ry(2.350741810354116) q[7];
rz(-1.2123754510001925) q[7];