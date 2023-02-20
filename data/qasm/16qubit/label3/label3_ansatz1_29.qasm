OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(2.98025406115189) q[0];
rz(-2.677206894596344) q[0];
ry(-3.1307415537639147) q[1];
rz(2.8834968964997274) q[1];
ry(-0.1454895774922029) q[2];
rz(0.015474507892424329) q[2];
ry(-1.5704182233231847) q[3];
rz(2.402635045483974) q[3];
ry(-1.5712637568047945) q[4];
rz(0.17352092786224738) q[4];
ry(-3.1399986654310448) q[5];
rz(-0.7518539427061083) q[5];
ry(-2.8777392821991548) q[6];
rz(0.06850955769982382) q[6];
ry(-0.08933462771652945) q[7];
rz(-1.090331821918958) q[7];
ry(3.138204122527237) q[8];
rz(-1.457458530880093) q[8];
ry(-0.6896648738123208) q[9];
rz(3.0473750062898333) q[9];
ry(-1.4624831183435392) q[10];
rz(-0.8182764876188403) q[10];
ry(-2.651211031524041) q[11];
rz(-0.34857530492329314) q[11];
ry(-2.5710603936877785) q[12];
rz(0.05316905114794543) q[12];
ry(0.5706856879079084) q[13];
rz(-2.8318540478906096) q[13];
ry(3.1240412190080615) q[14];
rz(-0.09507088862537307) q[14];
ry(2.2392905353602464) q[15];
rz(-3.0935815693555706) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.600153930801651) q[0];
rz(-2.979362786049203) q[0];
ry(-2.9120426975411835) q[1];
rz(3.1198597088944475) q[1];
ry(-1.571806526573425) q[2];
rz(-2.6295268671245435) q[2];
ry(0.6007836319335373) q[3];
rz(-2.111010185090092) q[3];
ry(-2.817987277937407) q[4];
rz(0.8580984992822168) q[4];
ry(-1.5711035749679292) q[5];
rz(2.7383113097327474) q[5];
ry(0.25630032479995307) q[6];
rz(-1.1744596419975188) q[6];
ry(2.2836075742593835) q[7];
rz(-0.07571764111336768) q[7];
ry(0.010801439524538166) q[8];
rz(-3.118180725514238) q[8];
ry(-0.4487843135794325) q[9];
rz(1.8651352846543734) q[9];
ry(-0.9559075424147663) q[10];
rz(0.21790914979698428) q[10];
ry(2.736330433912895) q[11];
rz(0.8384418115571525) q[11];
ry(0.5635163436719282) q[12];
rz(-2.7024192537499228) q[12];
ry(-1.3547906373398777) q[13];
rz(-1.7008678136021302) q[13];
ry(-0.12045252942431069) q[14];
rz(-3.119759351714961) q[14];
ry(1.3390351753210972) q[15];
rz(0.14329802256230717) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.6964279244614957) q[0];
rz(3.1366511842899167) q[0];
ry(1.5676445587726988) q[1];
rz(-2.8709427371838387) q[1];
ry(-3.1231059215436643) q[2];
rz(-0.9654022060701948) q[2];
ry(-0.21351802326513478) q[3];
rz(-3.113681396652609) q[3];
ry(-1.8760717548053976) q[4];
rz(0.9971628069714935) q[4];
ry(-1.5274752936011229) q[5];
rz(-1.4790089102295016) q[5];
ry(-1.5711869804359866) q[6];
rz(2.5154053385211212) q[6];
ry(0.9078466059041994) q[7];
rz(2.5154901884176617) q[7];
ry(-2.6375430832170808) q[8];
rz(2.2860399101587) q[8];
ry(2.8393841381607974) q[9];
rz(-0.41758990059565626) q[9];
ry(1.873309000349713) q[10];
rz(2.3420070437935467) q[10];
ry(-1.3301911211778883) q[11];
rz(0.8797214746368125) q[11];
ry(2.721455727525349) q[12];
rz(-1.7662909060880065) q[12];
ry(-1.034396428179531) q[13];
rz(0.18806069934258482) q[13];
ry(-0.15782770475061092) q[14];
rz(1.1087992518411518) q[14];
ry(0.818494507791844) q[15];
rz(-0.06133012010579546) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.5740264360917482) q[0];
rz(1.3221033602193855) q[0];
ry(-2.82646916894143) q[1];
rz(0.3094182060806073) q[1];
ry(-0.6001318598866483) q[2];
rz(0.3855363914593602) q[2];
ry(0.10051470221270653) q[3];
rz(1.5415301797222432) q[3];
ry(-0.5035887236102768) q[4];
rz(-1.946382908453797) q[4];
ry(-3.0283170489207247) q[5];
rz(1.241341007814239) q[5];
ry(-0.0006444785634385863) q[6];
rz(2.2979883675371693) q[6];
ry(3.134603452230864) q[7];
rz(-2.140710443007173) q[7];
ry(-3.0532425529259024) q[8];
rz(-0.7809837826562473) q[8];
ry(-3.082051093879998) q[9];
rz(1.9049688807226257) q[9];
ry(-1.5046032308954453) q[10];
rz(1.9117909157990183) q[10];
ry(1.1786868096477097) q[11];
rz(-0.9189941281195106) q[11];
ry(0.5449331480910861) q[12];
rz(0.5517526677637452) q[12];
ry(1.5760769908582386) q[13];
rz(2.37686308692711) q[13];
ry(0.06319054232029409) q[14];
rz(0.21779833219296577) q[14];
ry(1.6083403845310935) q[15];
rz(-1.724130606861908) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.7662593100704763) q[0];
rz(0.2821087029051233) q[0];
ry(-1.5983039071269627) q[1];
rz(1.4291635006632086) q[1];
ry(-0.6442130298825386) q[2];
rz(0.8702369337372361) q[2];
ry(-2.6240607117076076) q[3];
rz(-2.7296863927112534) q[3];
ry(0.5104127975249826) q[4];
rz(1.3032907221418093) q[4];
ry(-2.498566137539236) q[5];
rz(-2.7577751496459095) q[5];
ry(-3.1356138193154277) q[6];
rz(-0.9627571984112632) q[6];
ry(3.10849218958752) q[7];
rz(-2.8131338710604834) q[7];
ry(0.5826611062386629) q[8];
rz(-2.406402359758445) q[8];
ry(-2.5762995649515483) q[9];
rz(-1.8379535307832535) q[9];
ry(-2.7579393419867073) q[10];
rz(-0.9028680146894015) q[10];
ry(-2.7562575047998172) q[11];
rz(-0.6943063947004061) q[11];
ry(-0.12602606009611073) q[12];
rz(-1.004802839256305) q[12];
ry(0.01518262531652592) q[13];
rz(1.4099937979455701) q[13];
ry(-3.1253918848992277) q[14];
rz(2.313471527222808) q[14];
ry(2.158854783384406) q[15];
rz(-1.530569365280173) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.140661967753598) q[0];
rz(2.0195128608616004) q[0];
ry(-0.9780921548561964) q[1];
rz(-0.5101711506012396) q[1];
ry(-0.19157287051765873) q[2];
rz(-2.776473853857486) q[2];
ry(1.5279400857608412) q[3];
rz(-3.0558651330542346) q[3];
ry(2.044714282666666) q[4];
rz(-0.30934179376226023) q[4];
ry(-0.023550146465359134) q[5];
rz(-2.200937292591629) q[5];
ry(-0.4222027668398019) q[6];
rz(2.174951199932627) q[6];
ry(-1.5750546579736895) q[7];
rz(-1.5892386769245093e-05) q[7];
ry(-2.262919037859213) q[8];
rz(-3.139248573797213) q[8];
ry(1.8690837505312423) q[9];
rz(0.5523695876021011) q[9];
ry(1.2547444018154696) q[10];
rz(-1.4432445503588491) q[10];
ry(-1.0754219531808211) q[11];
rz(0.6292239230920348) q[11];
ry(1.1783228903010297) q[12];
rz(0.1676535533583509) q[12];
ry(2.4762834609432423) q[13];
rz(-0.5632988286290148) q[13];
ry(-3.0307466636259597) q[14];
rz(-1.6274711327693772) q[14];
ry(1.4052292517609422) q[15];
rz(2.201342326454842) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.902654028995766) q[0];
rz(-2.221117945445812) q[0];
ry(0.8993946765852694) q[1];
rz(-2.4694513281522603) q[1];
ry(-1.7494443378352877) q[2];
rz(1.5877310338124437) q[2];
ry(0.7955977349310244) q[3];
rz(-1.7788992650723152) q[3];
ry(0.9360789934400843) q[4];
rz(-0.37770586568259673) q[4];
ry(2.516426879448744) q[5];
rz(2.884898992397292) q[5];
ry(-0.018737971607325488) q[6];
rz(0.6551519889991162) q[6];
ry(-1.5852116904086628) q[7];
rz(0.01232686273421635) q[7];
ry(-1.572233246427197) q[8];
rz(3.1413829507825284) q[8];
ry(1.4763255212520425) q[9];
rz(-0.014198684318756705) q[9];
ry(-2.0993018767406424) q[10];
rz(2.870593631223313) q[10];
ry(0.11768512962810007) q[11];
rz(-1.5000971928388038) q[11];
ry(0.07565918737816059) q[12];
rz(-1.2203670088972824) q[12];
ry(2.944027938940644) q[13];
rz(0.8126057876241682) q[13];
ry(-3.0149199090850773) q[14];
rz(0.49326887091274674) q[14];
ry(2.8844284103116524) q[15];
rz(0.5760722532621887) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.04370455649143068) q[0];
rz(1.2263048856944747) q[0];
ry(2.894250691219089) q[1];
rz(1.1964362718090378) q[1];
ry(2.3921731164532254) q[2];
rz(-0.5463333667878443) q[2];
ry(2.8764057272663885) q[3];
rz(2.1168837277716896) q[3];
ry(2.2935689599129496) q[4];
rz(-0.8785741347583039) q[4];
ry(-1.8718649948425097) q[5];
rz(-1.7915487454799144) q[5];
ry(-1.5284031883439697) q[6];
rz(-3.1354613043859696) q[6];
ry(1.9709991589054425) q[7];
rz(3.0991216474867955) q[7];
ry(2.1434059123642344) q[8];
rz(0.5085344981105905) q[8];
ry(-1.5683016637487182) q[9];
rz(1.5625947379509164) q[9];
ry(-0.15813160393949846) q[10];
rz(-2.8305714339613206) q[10];
ry(0.8485084041729039) q[11];
rz(-0.5907287596055321) q[11];
ry(-2.4774564182657275) q[12];
rz(-1.90659051343794) q[12];
ry(0.9237308073742647) q[13];
rz(-0.46876623919952815) q[13];
ry(-2.0456445239440058) q[14];
rz(2.3266512739626926) q[14];
ry(1.6384430932091805) q[15];
rz(-1.2096403019391424) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.00912205827161241) q[0];
rz(1.8395742981023506) q[0];
ry(-3.0848404371038693) q[1];
rz(1.2428602698931597) q[1];
ry(1.4712160957182392) q[2];
rz(0.023230519608709758) q[2];
ry(1.0263060017677894) q[3];
rz(-2.644661182619446) q[3];
ry(-1.7191366723399637) q[4];
rz(1.3489616401208042) q[4];
ry(3.1364168374044437) q[5];
rz(-0.3153568593992624) q[5];
ry(-1.4809082598967125) q[6];
rz(2.1542777792440893) q[6];
ry(-2.613157557915709) q[7];
rz(1.7021241563336282) q[7];
ry(3.12071639870737) q[8];
rz(0.5079658715455166) q[8];
ry(1.5570594343108404) q[9];
rz(-2.2992206615265705) q[9];
ry(1.6297044822596103) q[10];
rz(-1.2788678654509522) q[10];
ry(-1.684696350279907) q[11];
rz(-2.127719718129071) q[11];
ry(-0.08257906487524913) q[12];
rz(0.028869327430358468) q[12];
ry(2.9911258496901696) q[13];
rz(-2.5451772511039428) q[13];
ry(3.1305975076544703) q[14];
rz(2.962188408385197) q[14];
ry(-0.03872265731474687) q[15];
rz(0.5810656266317247) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.0920914737138046) q[0];
rz(0.059378507035608596) q[0];
ry(0.8860182969781257) q[1];
rz(1.0765419403280525) q[1];
ry(-2.008375856485253) q[2];
rz(0.7141446026960298) q[2];
ry(2.9193218149046705) q[3];
rz(1.6563628212884414) q[3];
ry(0.2915259828578442) q[4];
rz(-1.1800571967474687) q[4];
ry(-2.772105372078265) q[5];
rz(-2.798057504893451) q[5];
ry(-3.13494867948362) q[6];
rz(-2.839898578756987) q[6];
ry(0.006763551335071783) q[7];
rz(1.414486002226732) q[7];
ry(1.5565127947024227) q[8];
rz(-2.8100777349001955) q[8];
ry(-0.006409182649867226) q[9];
rz(-1.5585448033579428) q[9];
ry(0.0028641231791493027) q[10];
rz(-0.9402567351128317) q[10];
ry(3.135205360390699) q[11];
rz(0.7698363155084725) q[11];
ry(-3.101403803546854) q[12];
rz(1.1572192646075825) q[12];
ry(-2.4597077269540217) q[13];
rz(-0.3769684233393802) q[13];
ry(1.9322550045481073) q[14];
rz(0.13369698139147232) q[14];
ry(1.3951569560894062) q[15];
rz(1.362723994955095) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.177939710738002) q[0];
rz(0.22990922970581296) q[0];
ry(0.04916014672865465) q[1];
rz(2.0340514499766242) q[1];
ry(3.054733256733422) q[2];
rz(0.6783714715563125) q[2];
ry(0.8415886960558732) q[3];
rz(2.9361058936464746) q[3];
ry(1.5058230693794494) q[4];
rz(-1.6110792262753524) q[4];
ry(3.1358051987713513) q[5];
rz(0.7180891249347764) q[5];
ry(1.6128803050351088) q[6];
rz(1.5324082699154538) q[6];
ry(1.5780740184165314) q[7];
rz(-2.9580279876104663) q[7];
ry(2.950974588666596) q[8];
rz(-2.75013924226824) q[8];
ry(-0.05381866808099911) q[9];
rz(1.5631412118791115) q[9];
ry(3.0910431533631364) q[10];
rz(1.5327548883107793) q[10];
ry(-1.520443958759365) q[11];
rz(0.9892621437260044) q[11];
ry(-0.06572055526134139) q[12];
rz(2.739651063057887) q[12];
ry(-0.1420640520693872) q[13];
rz(-1.1803422345575119) q[13];
ry(0.02540026340951851) q[14];
rz(0.04779054832694473) q[14];
ry(2.9110964819239977) q[15];
rz(-0.4781549923737849) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.0009423561017904047) q[0];
rz(2.1284476272440216) q[0];
ry(-2.2019629734656254) q[1];
rz(0.1771404898997764) q[1];
ry(1.7699199016927818) q[2];
rz(-1.105584150913111) q[2];
ry(0.2748750100173529) q[3];
rz(-1.8287090150526464) q[3];
ry(1.5411893015648044) q[4];
rz(-2.7477326279874394) q[4];
ry(-0.0026984906153973477) q[5];
rz(-1.800151040528718) q[5];
ry(0.5402250839888609) q[6];
rz(-2.9088260245311837) q[6];
ry(-3.059197421734303) q[7];
rz(-2.705534922153681) q[7];
ry(0.1170334604007701) q[8];
rz(-0.032278883230385524) q[8];
ry(0.07902637687788427) q[9];
rz(-0.816341398479708) q[9];
ry(-0.01883188555434767) q[10];
rz(-3.0486296229387864) q[10];
ry(-3.105804825795593) q[11];
rz(-2.6037942964631138) q[11];
ry(-1.3709603648731905) q[12];
rz(0.9249948885345667) q[12];
ry(-1.8709493045361978) q[13];
rz(2.308605017397592) q[13];
ry(2.9894819373236543) q[14];
rz(2.332139715621297) q[14];
ry(-0.4918560683649451) q[15];
rz(-0.634730150719822) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.5019507733814886) q[0];
rz(-1.6803215151048518) q[0];
ry(2.7059201931906114) q[1];
rz(-2.6136912400120833) q[1];
ry(1.8367717260934056) q[2];
rz(-0.6710029024545419) q[2];
ry(0.8476025749940991) q[3];
rz(0.7193200231857918) q[3];
ry(2.9251949031269366) q[4];
rz(-2.201729407962001) q[4];
ry(1.471367496557999) q[5];
rz(-1.4320282454726776) q[5];
ry(0.2957703958629363) q[6];
rz(-0.5474385590107326) q[6];
ry(0.0053121222075427355) q[7];
rz(2.776704108670967) q[7];
ry(0.24429573510614144) q[8];
rz(-2.3804304011454605) q[8];
ry(2.5145616967998694) q[9];
rz(1.6477933151170767) q[9];
ry(-0.04158848233604929) q[10];
rz(-0.594859882101428) q[10];
ry(-3.055351277212692) q[11];
rz(-2.7069994408267957) q[11];
ry(2.9903417038661595) q[12];
rz(-0.9656287122801918) q[12];
ry(-0.9651847111601493) q[13];
rz(2.6038280355615355) q[13];
ry(-2.4364288337924602) q[14];
rz(-2.9064871401500594) q[14];
ry(1.7398383162933992) q[15];
rz(-1.3869304434059828) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.141064027819658) q[0];
rz(-2.174862981890245) q[0];
ry(2.2782622853915657) q[1];
rz(0.13100649245513363) q[1];
ry(-2.8322634300288474) q[2];
rz(0.49613746633941647) q[2];
ry(-2.6183758183935266) q[3];
rz(2.6194184963801423) q[3];
ry(3.136042904726555) q[4];
rz(-2.0377491510756047) q[4];
ry(-0.16745909644841275) q[5];
rz(-1.8917004556743005) q[5];
ry(-1.5399948483390673) q[6];
rz(-2.524470813515267) q[6];
ry(-2.0647763904273413) q[7];
rz(2.208466175494396) q[7];
ry(-2.0262542190760175) q[8];
rz(1.833395446509868) q[8];
ry(0.03533961985217567) q[9];
rz(1.178147897328392) q[9];
ry(1.1939257804415693) q[10];
rz(-2.081390876140085) q[10];
ry(1.5286293483363984) q[11];
rz(-0.4008240298441246) q[11];
ry(-0.05495765085513238) q[12];
rz(2.939540715839631) q[12];
ry(0.04910090664831035) q[13];
rz(0.5423481502715991) q[13];
ry(3.0347770299976378) q[14];
rz(0.2122607780559047) q[14];
ry(2.8383095902372277) q[15];
rz(-2.5628553732328068) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.1547096904939664) q[0];
rz(-0.22323705852866207) q[0];
ry(-0.0010211556471526606) q[1];
rz(0.44668761162970144) q[1];
ry(2.7684956476608766) q[2];
rz(0.38360323756272985) q[2];
ry(0.7077852594445154) q[3];
rz(2.7997007272961203) q[3];
ry(2.6810221730864203) q[4];
rz(-1.9144901040216866) q[4];
ry(1.0755907393879403) q[5];
rz(1.7006995304093575) q[5];
ry(-3.1338069073821515) q[6];
rz(-0.9468069364698302) q[6];
ry(3.139408188254874) q[7];
rz(2.1282558984086615) q[7];
ry(3.0270297308867335) q[8];
rz(-1.0425537499053723) q[8];
ry(-0.004105184707676195) q[9];
rz(-1.9567651304058993) q[9];
ry(-0.028312834901435418) q[10];
rz(-1.4208693644350077) q[10];
ry(-0.0032526154898322446) q[11];
rz(1.0857056377554821) q[11];
ry(-3.1385123483870756) q[12];
rz(2.9882773955860125) q[12];
ry(1.0254996215619618) q[13];
rz(-2.340811826119524) q[13];
ry(-0.8022364870508092) q[14];
rz(0.23569327382421376) q[14];
ry(-2.8332155552062224) q[15];
rz(-2.57014245262132) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.1406374885650377) q[0];
rz(1.4017415833858775) q[0];
ry(-0.844420872083071) q[1];
rz(2.9025180597648803) q[1];
ry(2.658188414228477) q[2];
rz(0.9929563491691119) q[2];
ry(0.5747498892691617) q[3];
rz(2.99413523797631) q[3];
ry(0.025897322975604098) q[4];
rz(-2.0579292227857566) q[4];
ry(-3.0474785632856465) q[5];
rz(-3.080888044277642) q[5];
ry(-1.7294153383281432) q[6];
rz(-0.5699012706169253) q[6];
ry(-1.1532608825878878) q[7];
rz(0.09019854853650157) q[7];
ry(2.2226585919087256) q[8];
rz(-0.5688535860245549) q[8];
ry(0.30833076794438846) q[9];
rz(2.4572519256027094) q[9];
ry(2.2061979838324097) q[10];
rz(-1.3651765896014896) q[10];
ry(0.22668422385771256) q[11];
rz(-2.325138194339262) q[11];
ry(2.268389723746443) q[12];
rz(1.8940878973050144) q[12];
ry(-2.9184744809039698) q[13];
rz(1.2309024331081568) q[13];
ry(-1.100786573881753) q[14];
rz(-0.3879790307119093) q[14];
ry(1.8756909278590095) q[15];
rz(-0.4284316923683839) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.743348321204566) q[0];
rz(1.4157925532869964) q[0];
ry(0.9432302441401426) q[1];
rz(0.9215083944648229) q[1];
ry(-2.675465591650145) q[2];
rz(2.823784409083228) q[2];
ry(1.8916998514901069) q[3];
rz(0.8831343645006092) q[3];
ry(-2.8528808357690956) q[4];
rz(2.844533600301158) q[4];
ry(1.6002540786987458) q[5];
rz(1.793515340360853) q[5];
ry(0.03431485494934722) q[6];
rz(1.8987935962374733) q[6];
ry(2.123440127504033) q[7];
rz(-1.9781219754866228) q[7];
ry(0.10188851661006648) q[8];
rz(-2.5797451931888955) q[8];
ry(-3.1368428360683174) q[9];
rz(-2.724884821677669) q[9];
ry(0.018438519644451642) q[10];
rz(2.637684092692879) q[10];
ry(-3.139832550961012) q[11];
rz(1.004740940962776) q[11];
ry(0.0004982932769292958) q[12];
rz(-2.961945652458094) q[12];
ry(-0.4762272421706843) q[13];
rz(1.5337884026003872) q[13];
ry(1.2718491736279436) q[14];
rz(-2.786029186913867) q[14];
ry(-0.17158904079231263) q[15];
rz(-0.7819441513223139) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.0024698678717092193) q[0];
rz(2.354827794353637) q[0];
ry(-1.8144344272658977) q[1];
rz(-0.890383211782228) q[1];
ry(-2.801135312705189) q[2];
rz(-1.2641941890561732) q[2];
ry(2.135242681300195) q[3];
rz(0.013625006024884402) q[3];
ry(-2.700997211323074) q[4];
rz(-2.5600611816352905) q[4];
ry(0.01417327438571392) q[5];
rz(1.7763778007739701) q[5];
ry(0.49201783078142236) q[6];
rz(-0.48121146040073537) q[6];
ry(-0.001592582416841977) q[7];
rz(1.22788023821367) q[7];
ry(-3.1407853118561673) q[8];
rz(1.2945248682542445) q[8];
ry(1.4606483100071617) q[9];
rz(-1.9755819416538651) q[9];
ry(-1.3912087306674612) q[10];
rz(-1.421489635950857) q[10];
ry(0.29314000634294907) q[11];
rz(-2.589226290067587) q[11];
ry(-0.3382061588701847) q[12];
rz(-0.3096158735527975) q[12];
ry(-1.576751832013242) q[13];
rz(-0.12270254739677267) q[13];
ry(2.023108171309615) q[14];
rz(2.902173134707004) q[14];
ry(1.8059238945632545) q[15];
rz(0.4808562437566754) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.7546930986435099) q[0];
rz(-0.16986074489339487) q[0];
ry(0.8996373743291176) q[1];
rz(2.5467174671304975) q[1];
ry(-2.168349507459576) q[2];
rz(-0.8708848734577278) q[2];
ry(-2.1749333287138737) q[3];
rz(0.272160598022577) q[3];
ry(1.3885477586885138) q[4];
rz(-2.5642204980912333) q[4];
ry(3.087568348757499) q[5];
rz(1.27151101595423) q[5];
ry(-0.19262537888267595) q[6];
rz(2.3125332541523695) q[6];
ry(2.361403584214368) q[7];
rz(1.567916528835286) q[7];
ry(0.023443617481269996) q[8];
rz(0.9141219514826918) q[8];
ry(-3.0562968144158607) q[9];
rz(-0.21844128368370158) q[9];
ry(-1.113719739899865) q[10];
rz(-1.3666337242884063) q[10];
ry(-1.6670683676848024) q[11];
rz(-1.4747742602800615) q[11];
ry(1.3846700399143979) q[12];
rz(3.1222088620815174) q[12];
ry(-0.07269949793265518) q[13];
rz(0.9021277704707702) q[13];
ry(0.3060120385729704) q[14];
rz(1.007718860347369) q[14];
ry(2.2634707902667133) q[15];
rz(0.6660412906426868) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.0006065863906460641) q[0];
rz(-2.117356664635646) q[0];
ry(1.9840027562805922) q[1];
rz(-2.636938571244297) q[1];
ry(0.16356415210636843) q[2];
rz(-1.4187477586176251) q[2];
ry(0.004421988644411349) q[3];
rz(-3.068864981022203) q[3];
ry(0.0696962747688108) q[4];
rz(1.7192516939082587) q[4];
ry(0.07470315097627012) q[5];
rz(1.6558246383832902) q[5];
ry(0.15204896428608095) q[6];
rz(-2.354360644971053) q[6];
ry(3.1341591312423214) q[7];
rz(-0.3272898613338175) q[7];
ry(2.2328552994911868) q[8];
rz(-2.6996976202082443) q[8];
ry(0.36686649825135503) q[9];
rz(0.10563150330444478) q[9];
ry(0.023705421393407172) q[10];
rz(-0.5344056660077039) q[10];
ry(0.0014340238956185303) q[11];
rz(-1.239000258220095) q[11];
ry(-1.319288605442918) q[12];
rz(0.2534377663773889) q[12];
ry(-3.1318506732200806) q[13];
rz(2.5842995757286102) q[13];
ry(1.5706055668555088) q[14];
rz(2.626654944615898) q[14];
ry(0.9851352580463022) q[15];
rz(1.2270883712735212) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.421033311211114) q[0];
rz(1.3122178259736055) q[0];
ry(-0.957210067930109) q[1];
rz(-2.57914817982123) q[1];
ry(-1.6703926691168027) q[2];
rz(2.2838612488092767) q[2];
ry(-0.6282800558748737) q[3];
rz(0.7652514743527443) q[3];
ry(-2.566503540686745) q[4];
rz(2.676660948160695) q[4];
ry(-0.05942938316110245) q[5];
rz(-1.872409957150595) q[5];
ry(0.18010894381800224) q[6];
rz(-2.5466685034782297) q[6];
ry(-1.8025816508209602) q[7];
rz(-2.8321488826514125) q[7];
ry(3.1167429753278215) q[8];
rz(-2.7039860875879054) q[8];
ry(1.1205846686910705) q[9];
rz(-0.017067854160083936) q[9];
ry(0.9986305288133943) q[10];
rz(2.9522456751967443) q[10];
ry(-0.407122279707659) q[11];
rz(2.367548368518331) q[11];
ry(0.31324311675772315) q[12];
rz(-0.22417693637548822) q[12];
ry(0.006374558889292282) q[13];
rz(0.19883261871193444) q[13];
ry(2.512959866125385) q[14];
rz(-0.14825384043606837) q[14];
ry(1.190250708346138) q[15];
rz(-3.031325221533421) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.0006382863046097143) q[0];
rz(-1.6303580631592371) q[0];
ry(-2.0745211345093866) q[1];
rz(-2.4193032566714296) q[1];
ry(-1.9845368598292081) q[2];
rz(-2.3974044355647828) q[2];
ry(1.7415415763426596) q[3];
rz(-3.0825061021195084) q[3];
ry(-2.9753260458728548) q[4];
rz(-1.0230390180054894) q[4];
ry(-3.0649602122443933) q[5];
rz(-0.8893917177296758) q[5];
ry(0.28420220530010987) q[6];
rz(3.0853708625622893) q[6];
ry(-0.006402465448002381) q[7];
rz(-2.010775126891072) q[7];
ry(2.4273636832153125) q[8];
rz(0.015023397438524102) q[8];
ry(-1.369172196880454) q[9];
rz(3.120532435603034) q[9];
ry(0.0351329353922738) q[10];
rz(-1.9720642249694127) q[10];
ry(-0.0046606165109608355) q[11];
rz(1.7631960168785632) q[11];
ry(-0.26690089527543204) q[12];
rz(0.5603726032774707) q[12];
ry(3.1313349418981575) q[13];
rz(0.2922086636994398) q[13];
ry(2.9396907539554546) q[14];
rz(-1.481553714390535) q[14];
ry(-2.907428786442777) q[15];
rz(-1.9628776170518236) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.7991616719341652) q[0];
rz(1.4735885614015238) q[0];
ry(-3.075854940181367) q[1];
rz(-3.0796469765666097) q[1];
ry(0.06231236482717248) q[2];
rz(-1.5721582974628205) q[2];
ry(-1.431980214125303) q[3];
rz(-0.27178450093989687) q[3];
ry(1.5322274373174762) q[4];
rz(0.27003321296258287) q[4];
ry(3.13292880570658) q[5];
rz(-1.5439493844043142) q[5];
ry(1.2115028177930371) q[6];
rz(-2.9469506164706774) q[6];
ry(0.9875042193605243) q[7];
rz(0.12793981391362497) q[7];
ry(1.466569410261397) q[8];
rz(-1.3603845927964815) q[8];
ry(0.6308308102588125) q[9];
rz(0.0521165438722901) q[9];
ry(-0.28318034337989956) q[10];
rz(-0.2682099997035267) q[10];
ry(-2.396338812294545) q[11];
rz(0.9963730784880135) q[11];
ry(0.13819764131881143) q[12];
rz(0.851333604071798) q[12];
ry(0.005911157743123674) q[13];
rz(0.7724808578229437) q[13];
ry(1.5439797708314014) q[14];
rz(-0.20469393334621785) q[14];
ry(-1.96028230009546) q[15];
rz(-1.2718512404804523) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.1308527296866164) q[0];
rz(1.929322937915824) q[0];
ry(-0.70306363208071) q[1];
rz(-1.2513602042854188) q[1];
ry(-3.0029125781257715) q[2];
rz(0.8315249709017324) q[2];
ry(-1.7344354493370782) q[3];
rz(2.377312683525192) q[3];
ry(-3.052633640832986) q[4];
rz(-2.8932049143412364) q[4];
ry(0.0006552584032173756) q[5];
rz(-0.5218267194638274) q[5];
ry(-0.524726427327886) q[6];
rz(3.057087163053727) q[6];
ry(-0.6743153575513063) q[7];
rz(-3.1138362676635314) q[7];
ry(0.44866971966629077) q[8];
rz(-1.1869263062446997) q[8];
ry(-1.4412674354139534) q[9];
rz(-2.9337825726494913) q[9];
ry(0.034060676953200364) q[10];
rz(-1.7830827897036345) q[10];
ry(1.8309917528888109) q[11];
rz(0.766791330888581) q[11];
ry(0.0008420364672963033) q[12];
rz(2.362199433512065) q[12];
ry(3.1301102567915557) q[13];
rz(2.5426053772538584) q[13];
ry(-0.31711131076221943) q[14];
rz(-0.5957161526412866) q[14];
ry(1.1312790979314018) q[15];
rz(1.5531897716886716) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.725224307918089) q[0];
rz(-2.376125423391912) q[0];
ry(-0.16999985919907076) q[1];
rz(-1.00940065904332) q[1];
ry(-1.6034158616222784) q[2];
rz(1.4674815209695387) q[2];
ry(2.936511770823895) q[3];
rz(2.3895507858417293) q[3];
ry(1.7029957454177271) q[4];
rz(-1.3760230214996054) q[4];
ry(-3.1274492607070408) q[5];
rz(0.006798180714143953) q[5];
ry(-0.020738972911257925) q[6];
rz(-3.0670211061753547) q[6];
ry(-2.3769338972485143) q[7];
rz(0.02521895578140576) q[7];
ry(-0.008171609453965086) q[8];
rz(2.3658363653223127) q[8];
ry(0.020291533145073374) q[9];
rz(-0.2021876954446835) q[9];
ry(-3.1380098514388965) q[10];
rz(-1.6824598555712438) q[10];
ry(-1.3842460169197741) q[11];
rz(-1.8273817952148639) q[11];
ry(3.140887639510865) q[12];
rz(-0.46486003641995577) q[12];
ry(0.0005199609790009774) q[13];
rz(0.30541048882405963) q[13];
ry(-1.7518149816180661) q[14];
rz(-2.139535384861106) q[14];
ry(-2.866573759083688) q[15];
rz(-2.4316305092333543) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.572023996662808) q[0];
rz(1.5617988195618084) q[0];
ry(-0.4978273623568251) q[1];
rz(-2.7808138938334643) q[1];
ry(1.366594602520848) q[2];
rz(0.24046364276760832) q[2];
ry(-1.5615675618030886) q[3];
rz(2.9996471582392816) q[3];
ry(-0.16063481995980486) q[4];
rz(-1.0969917203790134) q[4];
ry(0.06544864679945625) q[5];
rz(0.13322612448032253) q[5];
ry(-2.249058931860211) q[6];
rz(2.741427856176158) q[6];
ry(-0.6795779789560098) q[7];
rz(-3.1312954048283514) q[7];
ry(-1.2053463002882712) q[8];
rz(1.8482173293555562) q[8];
ry(0.25300654829258723) q[9];
rz(2.0120085243266104) q[9];
ry(0.37692003840096694) q[10];
rz(-1.1938198468506416) q[10];
ry(0.649696401499941) q[11];
rz(-2.903861226465387) q[11];
ry(-0.015467022026129503) q[12];
rz(-2.185519215354846) q[12];
ry(1.5685273328105038) q[13];
rz(1.2598878587105535) q[13];
ry(1.0744423386942117) q[14];
rz(-0.2708094114873294) q[14];
ry(-0.8060321387145262) q[15];
rz(0.9523491996991322) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.571588472630107) q[0];
rz(-0.8679323473838088) q[0];
ry(1.5733071236520537) q[1];
rz(1.57088763483769) q[1];
ry(1.756131579645365) q[2];
rz(-2.629871931916795) q[2];
ry(-0.008561565439084262) q[3];
rz(2.4729940932018066) q[3];
ry(-0.6074882117533864) q[4];
rz(-0.428595949806394) q[4];
ry(0.021014953950885484) q[5];
rz(-2.83911885767474) q[5];
ry(-1.5713856943005473) q[6];
rz(3.0236370348981096) q[6];
ry(2.384570291476315) q[7];
rz(-2.0283982700885583) q[7];
ry(3.1309763235652897) q[8];
rz(-0.7231574864644426) q[8];
ry(0.005782808144507712) q[9];
rz(1.0122952358641795) q[9];
ry(-3.138203895970428) q[10];
rz(-1.9958579449616367) q[10];
ry(-1.144453997077984) q[11];
rz(-1.723179016912595) q[11];
ry(1.5861393396268475) q[12];
rz(-1.5706287183873808) q[12];
ry(-6.764509971635135e-05) q[13];
rz(-0.507452407692802) q[13];
ry(-1.5625373298496612) q[14];
rz(-3.128134967696147) q[14];
ry(1.68976476932999) q[15];
rz(1.8383718318850473) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.5714190723745363) q[0];
rz(1.5698137353308503) q[0];
ry(-0.8993477024134071) q[1];
rz(3.141504125732882) q[1];
ry(3.140583279781004) q[2];
rz(2.0971731803859135) q[2];
ry(-2.283870448796398) q[3];
rz(1.093762491832901) q[3];
ry(0.8756621510576884) q[4];
rz(1.3175192770893205) q[4];
ry(-0.07992311000089533) q[5];
rz(0.6922948244278331) q[5];
ry(-0.07768995159300651) q[6];
rz(0.4183071557810924) q[6];
ry(-0.013749412171706022) q[7];
rz(-1.6935596785706966) q[7];
ry(0.8334263734435167) q[8];
rz(-1.4476985288246302) q[8];
ry(2.60117991690872) q[9];
rz(-3.1251110048487174) q[9];
ry(3.1299079774077847) q[10];
rz(1.0738790288640079) q[10];
ry(0.023217661124781408) q[11];
rz(0.07216735194887026) q[11];
ry(-1.571908191209568) q[12];
rz(-1.5745065218919683) q[12];
ry(3.139299912513598) q[13];
rz(0.7548409428585452) q[13];
ry(-1.3830868853380736) q[14];
rz(-1.5761293150556615) q[14];
ry(-1.61816397801066) q[15];
rz(-0.0016285338323686136) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.566531119890076) q[0];
rz(3.1388669422773083) q[0];
ry(-1.5722033889846037) q[1];
rz(-2.4967635862624182) q[1];
ry(0.015372772484901289) q[2];
rz(-1.8785791152150724) q[2];
ry(3.0414654047367873) q[3];
rz(-2.1323293856436267) q[3];
ry(3.1343381173194143) q[4];
rz(0.24850370933022742) q[4];
ry(3.134336626836571) q[5];
rz(-2.079772112771332) q[5];
ry(-0.6703810241430089) q[6];
rz(1.6159742994324524) q[6];
ry(3.029775973902089) q[7];
rz(0.3984859986032551) q[7];
ry(3.1247070010602016) q[8];
rz(-0.9826433604318667) q[8];
ry(1.7820181899704393) q[9];
rz(1.6001368240728633) q[9];
ry(-3.1411041805651894) q[10];
rz(-1.3697909253029976) q[10];
ry(-1.5703359005552437) q[11];
rz(-1.5707485697671004) q[11];
ry(1.5788697720723057) q[12];
rz(3.041954771012075) q[12];
ry(-1.5710284397730874) q[13];
rz(0.049559234088720386) q[13];
ry(1.5589929801579387) q[14];
rz(-3.1023085909717074) q[14];
ry(-2.8407626755696493) q[15];
rz(-3.138809290035159) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.569781995497311) q[0];
rz(-2.1768095728087085) q[0];
ry(-1.8125507059753572) q[1];
rz(1.204181123261191) q[1];
ry(-0.10520260236599703) q[2];
rz(0.5916218626633879) q[2];
ry(2.2720852519285013) q[3];
rz(-0.9710497326924202) q[3];
ry(-1.3644096840148858) q[4];
rz(-0.9217823902054426) q[4];
ry(-3.0428321406496024) q[5];
rz(0.7963290459159723) q[5];
ry(-1.6090175505656052) q[6];
rz(1.9641430521735446) q[6];
ry(-3.1215956515294914) q[7];
rz(2.558704928923984) q[7];
ry(0.09301844596723985) q[8];
rz(1.8864867065562179) q[8];
ry(1.5962269789624797) q[9];
rz(2.2196402451540447) q[9];
ry(-3.1388115512374) q[10];
rz(1.5299354934967195) q[10];
ry(-1.3615252338362884) q[11];
rz(-2.5894426232522294) q[11];
ry(-0.000305707842703562) q[12];
rz(-0.020223607753758602) q[12];
ry(-3.076475088500937) q[13];
rz(1.619130071044334) q[13];
ry(-1.5716979529198598) q[14];
rz(2.0569317951540693) q[14];
ry(-1.3937541248494192) q[15];
rz(1.5734369360313405) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.035465700700941) q[0];
rz(1.0541614465084557) q[0];
ry(1.0280716469284297) q[1];
rz(2.9762431418868727) q[1];
ry(2.374550217459375) q[2];
rz(1.8168404315495703) q[2];
ry(0.044850947816877866) q[3];
rz(-1.103167821161134) q[3];
ry(1.8800652369807942) q[4];
rz(0.12614429491079537) q[4];
ry(-3.1175834956329256) q[5];
rz(-2.861623460225837) q[5];
ry(1.3592544678575882) q[6];
rz(-1.1460182062727513) q[6];
ry(1.6296818968478668) q[7];
rz(-1.0157443352868123) q[7];
ry(0.0006293914345434359) q[8];
rz(-1.8627287020206085) q[8];
ry(-1.698330029476938) q[9];
rz(-1.5953696944437006) q[9];
ry(-1.566114756585951) q[10];
rz(1.7398944825818048) q[10];
ry(2.891641626805394) q[11];
rz(-1.0482864676819652) q[11];
ry(0.43745623019165364) q[12];
rz(2.5973804364603508) q[12];
ry(1.5714135859351614) q[13];
rz(-2.223671480867803) q[13];
ry(3.0399809985716972) q[14];
rz(0.491422164417532) q[14];
ry(1.5714055877385995) q[15];
rz(0.9176027176872884) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.1398167771589067) q[0];
rz(2.2362642071042367) q[0];
ry(-1.52794583049148) q[1];
rz(-1.529661033578339) q[1];
ry(-2.942745822985074) q[2];
rz(3.1322675821428176) q[2];
ry(3.014004767243559) q[3];
rz(0.37938760232335245) q[3];
ry(-1.264853595812972) q[4];
rz(2.8040047654136906) q[4];
ry(0.03942390373110172) q[5];
rz(2.9026137888423285) q[5];
ry(-1.5753022444566327) q[6];
rz(0.020090582057634306) q[6];
ry(-0.003166526890631805) q[7];
rz(-0.5235248785268933) q[7];
ry(0.008236669420400484) q[8];
rz(1.490379452792707) q[8];
ry(-1.5723228041163795) q[9];
rz(1.5402640786916963) q[9];
ry(-3.129628242784943) q[10];
rz(-1.110427700481214) q[10];
ry(0.009074540900119388) q[11];
rz(-2.105732011519642) q[11];
ry(3.1410489832576163) q[12];
rz(-2.0171120273603345) q[12];
ry(-3.140371668122344) q[13];
rz(-2.672493532427546) q[13];
ry(-1.4484176226085568) q[14];
rz(-3.1069701256289015) q[14];
ry(0.0011863575677697469) q[15];
rz(-0.8962590404529892) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.222819426932028) q[0];
rz(0.1758225440519299) q[0];
ry(1.4211883534492469) q[1];
rz(-1.6194413972710224) q[1];
ry(-3.1059071467948596) q[2];
rz(1.4763672240066412) q[2];
ry(-0.1496275659374282) q[3];
rz(-0.10520456312611781) q[3];
ry(-0.5947271468301336) q[4];
rz(-2.20083310049003) q[4];
ry(-3.0300441818226425) q[5];
rz(-1.1441775687884705) q[5];
ry(-0.9416837661895833) q[6];
rz(0.1805268721239472) q[6];
ry(1.5982005710785356) q[7];
rz(-1.451253066056355) q[7];
ry(-1.5288409447677207) q[8];
rz(-1.410126756022315) q[8];
ry(-1.5798821566867423) q[9];
rz(0.15456555701135768) q[9];
ry(2.9911572959505937) q[10];
rz(-1.1181327174126352) q[10];
ry(2.8556797389065744) q[11];
rz(1.1478732455308434) q[11];
ry(-1.57385830105716) q[12];
rz(2.8675945522689195) q[12];
ry(3.093817250849583) q[13];
rz(1.24555547733395) q[13];
ry(1.7016402769681278) q[14];
rz(-1.4013232538922473) q[14];
ry(1.6136212520384998) q[15];
rz(-1.4257675095053364) q[15];