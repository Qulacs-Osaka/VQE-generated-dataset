OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.2828474776474428) q[0];
ry(-3.065119169881998) q[1];
cx q[0],q[1];
ry(1.4956109055512385) q[0];
ry(0.15318047677023205) q[1];
cx q[0],q[1];
ry(1.7705686395465294) q[2];
ry(-0.36594728219462824) q[3];
cx q[2],q[3];
ry(-0.2032772829456082) q[2];
ry(-1.2613451264319102) q[3];
cx q[2],q[3];
ry(-1.3537223325309846) q[4];
ry(-2.7510803144097595) q[5];
cx q[4],q[5];
ry(-0.4472579016515441) q[4];
ry(-1.5963405096561933) q[5];
cx q[4],q[5];
ry(2.6523164132025907) q[6];
ry(-2.73284989556477) q[7];
cx q[6],q[7];
ry(-1.9856194453954108) q[6];
ry(1.5774057855983503) q[7];
cx q[6],q[7];
ry(1.6352126336956008) q[0];
ry(-1.6517111125600508) q[2];
cx q[0],q[2];
ry(2.7146471918155823) q[0];
ry(-2.5217109806248836) q[2];
cx q[0],q[2];
ry(-0.7608666346923654) q[2];
ry(-0.34930930733320315) q[4];
cx q[2],q[4];
ry(1.998728635753074) q[2];
ry(0.6433874101171914) q[4];
cx q[2],q[4];
ry(2.7253161249797206) q[4];
ry(-1.2406401587550624) q[6];
cx q[4],q[6];
ry(-1.8474962615661932) q[4];
ry(0.2009397552280081) q[6];
cx q[4],q[6];
ry(0.38730491392730365) q[1];
ry(0.14865205683159832) q[3];
cx q[1],q[3];
ry(1.0818526414163903) q[1];
ry(2.2831940377324162) q[3];
cx q[1],q[3];
ry(-1.3016471479017575) q[3];
ry(-0.4890899790188685) q[5];
cx q[3],q[5];
ry(0.6518410238534775) q[3];
ry(3.040946116248847) q[5];
cx q[3],q[5];
ry(-2.0560867295678804) q[5];
ry(-1.3122890491088564) q[7];
cx q[5],q[7];
ry(-0.8885913749725951) q[5];
ry(1.96467089467789) q[7];
cx q[5],q[7];
ry(-0.14489074203976018) q[0];
ry(1.8953532757154925) q[3];
cx q[0],q[3];
ry(-1.2030450601041627) q[0];
ry(1.6670507941960517) q[3];
cx q[0],q[3];
ry(1.2577441994543657) q[1];
ry(1.1816666917154144) q[2];
cx q[1],q[2];
ry(0.7795885934692897) q[1];
ry(-0.5572128569769619) q[2];
cx q[1],q[2];
ry(0.18168140537474198) q[2];
ry(0.4705948613565871) q[5];
cx q[2],q[5];
ry(-0.887953166956998) q[2];
ry(1.0658522657061074) q[5];
cx q[2],q[5];
ry(2.668154805421784) q[3];
ry(-2.236807984696771) q[4];
cx q[3],q[4];
ry(-2.2450646070701237) q[3];
ry(0.32908777839346653) q[4];
cx q[3],q[4];
ry(-3.1225258251161585) q[4];
ry(-1.561549622015538) q[7];
cx q[4],q[7];
ry(-2.4958959873314734) q[4];
ry(2.5332619770054032) q[7];
cx q[4],q[7];
ry(-0.7041905460357666) q[5];
ry(0.9566703819139609) q[6];
cx q[5],q[6];
ry(0.04865846112499295) q[5];
ry(2.351395580000797) q[6];
cx q[5],q[6];
ry(-0.3310498751956842) q[0];
ry(1.2448296818852935) q[1];
cx q[0],q[1];
ry(1.8451547070118208) q[0];
ry(1.4293914993591075) q[1];
cx q[0],q[1];
ry(1.565899926097418) q[2];
ry(-0.8379155152120896) q[3];
cx q[2],q[3];
ry(1.4395210799155027) q[2];
ry(-0.6422714233224625) q[3];
cx q[2],q[3];
ry(1.9045464084291897) q[4];
ry(-1.9610798496454578) q[5];
cx q[4],q[5];
ry(-1.564085818185995) q[4];
ry(2.307596892914763) q[5];
cx q[4],q[5];
ry(-1.3556991497677) q[6];
ry(2.7067210053259103) q[7];
cx q[6],q[7];
ry(0.024897465748839755) q[6];
ry(0.5536190799014935) q[7];
cx q[6],q[7];
ry(1.8857115446272998) q[0];
ry(3.0108561144721064) q[2];
cx q[0],q[2];
ry(1.0415711938764947) q[0];
ry(-0.8902495348690032) q[2];
cx q[0],q[2];
ry(-0.4181307691737546) q[2];
ry(-0.8531710380035555) q[4];
cx q[2],q[4];
ry(0.5736061766291787) q[2];
ry(0.23383467670720112) q[4];
cx q[2],q[4];
ry(-2.802881828471411) q[4];
ry(-0.1956330179978405) q[6];
cx q[4],q[6];
ry(-1.9380803384776968) q[4];
ry(2.248990986607711) q[6];
cx q[4],q[6];
ry(0.7739411882260089) q[1];
ry(-0.5433487971192282) q[3];
cx q[1],q[3];
ry(1.4417302335050337) q[1];
ry(-0.10630283974202667) q[3];
cx q[1],q[3];
ry(-1.0761118043437312) q[3];
ry(0.06109529184762863) q[5];
cx q[3],q[5];
ry(0.40543898023891) q[3];
ry(1.2737537512791892) q[5];
cx q[3],q[5];
ry(0.7364185841101297) q[5];
ry(-0.03150097957147757) q[7];
cx q[5],q[7];
ry(-2.133620240007949) q[5];
ry(0.4135431506302796) q[7];
cx q[5],q[7];
ry(-2.4647685771905263) q[0];
ry(1.695020530355247) q[3];
cx q[0],q[3];
ry(2.0242010752421686) q[0];
ry(0.5429313693601627) q[3];
cx q[0],q[3];
ry(2.4671601450905922) q[1];
ry(-0.7668104575651409) q[2];
cx q[1],q[2];
ry(-0.8108722986574302) q[1];
ry(-2.97465672688081) q[2];
cx q[1],q[2];
ry(-2.4684136641553387) q[2];
ry(1.0642286182204228) q[5];
cx q[2],q[5];
ry(1.2799573155396073) q[2];
ry(-1.8973148804928295) q[5];
cx q[2],q[5];
ry(-1.0162772416616201) q[3];
ry(1.7894888381881868) q[4];
cx q[3],q[4];
ry(-0.730108528403977) q[3];
ry(3.0918665079913645) q[4];
cx q[3],q[4];
ry(-1.1911665280507004) q[4];
ry(0.10373385487805997) q[7];
cx q[4],q[7];
ry(-1.7732973758286104) q[4];
ry(0.22941148739627193) q[7];
cx q[4],q[7];
ry(1.765159552791843) q[5];
ry(-2.0973460211280877) q[6];
cx q[5],q[6];
ry(2.574816131246375) q[5];
ry(1.75827720655244) q[6];
cx q[5],q[6];
ry(-1.249638822857472) q[0];
ry(1.3167781344901224) q[1];
cx q[0],q[1];
ry(-0.4969476161562577) q[0];
ry(0.2635922497085179) q[1];
cx q[0],q[1];
ry(2.0046584303042287) q[2];
ry(1.5897841634843342) q[3];
cx q[2],q[3];
ry(2.1107541857033345) q[2];
ry(1.2154690206647136) q[3];
cx q[2],q[3];
ry(-0.8901835530113553) q[4];
ry(-1.0319746146318476) q[5];
cx q[4],q[5];
ry(2.6713057094732156) q[4];
ry(-3.073155302311444) q[5];
cx q[4],q[5];
ry(-1.407426745034403) q[6];
ry(1.363298997947891) q[7];
cx q[6],q[7];
ry(-1.3556811044516914) q[6];
ry(1.8122131673493067) q[7];
cx q[6],q[7];
ry(-2.2442750604956245) q[0];
ry(-1.2740624115986943) q[2];
cx q[0],q[2];
ry(1.609905399989549) q[0];
ry(-2.731995813612783) q[2];
cx q[0],q[2];
ry(-0.3000478499644055) q[2];
ry(-0.4622162305699473) q[4];
cx q[2],q[4];
ry(-0.29238807948157064) q[2];
ry(-2.287524143772599) q[4];
cx q[2],q[4];
ry(-1.6551392715234572) q[4];
ry(-3.0912787495204066) q[6];
cx q[4],q[6];
ry(-3.0883865677748292) q[4];
ry(-0.8892710896526577) q[6];
cx q[4],q[6];
ry(-1.9846830277310206) q[1];
ry(-1.043157329671672) q[3];
cx q[1],q[3];
ry(1.1897422550760486) q[1];
ry(-0.9303398600831885) q[3];
cx q[1],q[3];
ry(0.1349123272587549) q[3];
ry(-0.8240668212051042) q[5];
cx q[3],q[5];
ry(-0.48943432615502225) q[3];
ry(-2.581976671588652) q[5];
cx q[3],q[5];
ry(2.996446557460857) q[5];
ry(-0.7691156159304179) q[7];
cx q[5],q[7];
ry(2.055868574489627) q[5];
ry(-0.4929426965114186) q[7];
cx q[5],q[7];
ry(-0.5283686490742596) q[0];
ry(-0.6573295829995192) q[3];
cx q[0],q[3];
ry(1.2597124138497684) q[0];
ry(-0.487091770040429) q[3];
cx q[0],q[3];
ry(-0.10987690240915882) q[1];
ry(2.1544293656334874) q[2];
cx q[1],q[2];
ry(-2.7294357516068595) q[1];
ry(-0.8778138244756017) q[2];
cx q[1],q[2];
ry(-2.6282698605544708) q[2];
ry(2.2412618429687923) q[5];
cx q[2],q[5];
ry(-1.2648085612825568) q[2];
ry(1.6648081338778882) q[5];
cx q[2],q[5];
ry(2.365283084948022) q[3];
ry(-2.944321539644636) q[4];
cx q[3],q[4];
ry(1.848045703205833) q[3];
ry(0.9127220319037748) q[4];
cx q[3],q[4];
ry(1.3249498769489962) q[4];
ry(-0.7853215684307028) q[7];
cx q[4],q[7];
ry(-1.0715259345387047) q[4];
ry(0.1360303317895708) q[7];
cx q[4],q[7];
ry(2.551134497742385) q[5];
ry(2.004690385636142) q[6];
cx q[5],q[6];
ry(0.22837173455271034) q[5];
ry(-1.2747664880254819) q[6];
cx q[5],q[6];
ry(-0.670572675940913) q[0];
ry(-0.6782311650680484) q[1];
cx q[0],q[1];
ry(2.2425666957188652) q[0];
ry(-1.2578944609614464) q[1];
cx q[0],q[1];
ry(2.1978546916183888) q[2];
ry(-2.083792101372098) q[3];
cx q[2],q[3];
ry(0.49379906283115993) q[2];
ry(-1.451806871094973) q[3];
cx q[2],q[3];
ry(1.792866800413492) q[4];
ry(0.45717926988805574) q[5];
cx q[4],q[5];
ry(-0.4321728043964379) q[4];
ry(2.2140686314544253) q[5];
cx q[4],q[5];
ry(0.6077606945131606) q[6];
ry(0.16757574031770225) q[7];
cx q[6],q[7];
ry(3.073706114099888) q[6];
ry(-2.165678778356204) q[7];
cx q[6],q[7];
ry(0.565770244687184) q[0];
ry(2.2568061788868494) q[2];
cx q[0],q[2];
ry(1.8481758847486258) q[0];
ry(1.7182397030821939) q[2];
cx q[0],q[2];
ry(1.5843383957081834) q[2];
ry(0.2240193420578855) q[4];
cx q[2],q[4];
ry(2.4206622262288096) q[2];
ry(-1.4566235124985718) q[4];
cx q[2],q[4];
ry(-2.662808220754142) q[4];
ry(-0.4999083160166834) q[6];
cx q[4],q[6];
ry(-2.409608764157907) q[4];
ry(-1.6403413880555533) q[6];
cx q[4],q[6];
ry(-1.0658666290982102) q[1];
ry(3.080173816905097) q[3];
cx q[1],q[3];
ry(-2.4875390585528994) q[1];
ry(-1.2926259433006537) q[3];
cx q[1],q[3];
ry(-2.6328131373595136) q[3];
ry(-0.2612020051285491) q[5];
cx q[3],q[5];
ry(-1.4389712807092598) q[3];
ry(-3.051941941408514) q[5];
cx q[3],q[5];
ry(-1.5894425400706842) q[5];
ry(1.8380842581680348) q[7];
cx q[5],q[7];
ry(2.6703471671477184) q[5];
ry(-1.9568839583415196) q[7];
cx q[5],q[7];
ry(-0.5615651956983243) q[0];
ry(-1.841789711021053) q[3];
cx q[0],q[3];
ry(1.7044728813320629) q[0];
ry(-2.9234010909454238) q[3];
cx q[0],q[3];
ry(-0.7502355921971677) q[1];
ry(-2.4356619932841896) q[2];
cx q[1],q[2];
ry(1.4803435802152356) q[1];
ry(1.6643431566287692) q[2];
cx q[1],q[2];
ry(0.324621930970829) q[2];
ry(-2.7092961776551094) q[5];
cx q[2],q[5];
ry(2.751769422112501) q[2];
ry(-1.9626973127793095) q[5];
cx q[2],q[5];
ry(-2.924505091389827) q[3];
ry(1.3092833229183327) q[4];
cx q[3],q[4];
ry(-1.8755201429703814) q[3];
ry(-0.8435386196498396) q[4];
cx q[3],q[4];
ry(-2.4340033664959977) q[4];
ry(2.8717861333023995) q[7];
cx q[4],q[7];
ry(-2.1998414251002116) q[4];
ry(-0.04115881807920729) q[7];
cx q[4],q[7];
ry(-0.7321939232981858) q[5];
ry(2.0678379579866606) q[6];
cx q[5],q[6];
ry(2.174040645270948) q[5];
ry(-2.388189125784846) q[6];
cx q[5],q[6];
ry(-0.11982058159369835) q[0];
ry(-1.4240733726155304) q[1];
cx q[0],q[1];
ry(0.7032525752497348) q[0];
ry(-2.2181034739698973) q[1];
cx q[0],q[1];
ry(0.2036692069766251) q[2];
ry(-2.5597476693139507) q[3];
cx q[2],q[3];
ry(0.6362563934026202) q[2];
ry(1.5131221642045058) q[3];
cx q[2],q[3];
ry(-0.7984131035180994) q[4];
ry(-1.3652593469028922) q[5];
cx q[4],q[5];
ry(1.0131507336382484) q[4];
ry(2.8797559357546967) q[5];
cx q[4],q[5];
ry(1.9991873270652505) q[6];
ry(-0.013008636821903521) q[7];
cx q[6],q[7];
ry(-1.0348946934389212) q[6];
ry(-1.5200779256105754) q[7];
cx q[6],q[7];
ry(-2.2343228382740055) q[0];
ry(1.6653939092746393) q[2];
cx q[0],q[2];
ry(-1.799083310603768) q[0];
ry(-0.20209613636999446) q[2];
cx q[0],q[2];
ry(-0.021801712039639903) q[2];
ry(-0.9495436284963101) q[4];
cx q[2],q[4];
ry(-0.8893124334409795) q[2];
ry(2.910004420720706) q[4];
cx q[2],q[4];
ry(1.992385242688032) q[4];
ry(-1.250748975548423) q[6];
cx q[4],q[6];
ry(-1.5796546644573297) q[4];
ry(-3.1158492809515663) q[6];
cx q[4],q[6];
ry(0.10922755751322377) q[1];
ry(1.9213445813484222) q[3];
cx q[1],q[3];
ry(1.058035865584893) q[1];
ry(-0.7894546912304552) q[3];
cx q[1],q[3];
ry(0.377332418733247) q[3];
ry(-0.8863301578853338) q[5];
cx q[3],q[5];
ry(-1.2488689225432328) q[3];
ry(-0.6683910367982273) q[5];
cx q[3],q[5];
ry(0.4717079559226596) q[5];
ry(-2.6215208723844787) q[7];
cx q[5],q[7];
ry(-0.5285676713184868) q[5];
ry(-1.689138776931146) q[7];
cx q[5],q[7];
ry(1.303249148540095) q[0];
ry(-1.2294394996785112) q[3];
cx q[0],q[3];
ry(-2.168507612112123) q[0];
ry(-2.8685640453799826) q[3];
cx q[0],q[3];
ry(-1.7075268106380481) q[1];
ry(0.47031216127044306) q[2];
cx q[1],q[2];
ry(2.3242388081261476) q[1];
ry(2.775507294715547) q[2];
cx q[1],q[2];
ry(-1.0750214630499233) q[2];
ry(1.3234365591792994) q[5];
cx q[2],q[5];
ry(-0.5019971828469796) q[2];
ry(-1.803094814465385) q[5];
cx q[2],q[5];
ry(1.5116900427050108) q[3];
ry(0.721285136353349) q[4];
cx q[3],q[4];
ry(2.793628793391522) q[3];
ry(0.27319806300292626) q[4];
cx q[3],q[4];
ry(-0.9712690325366049) q[4];
ry(2.7380301863660517) q[7];
cx q[4],q[7];
ry(2.4126713135485374) q[4];
ry(-2.5268110802873394) q[7];
cx q[4],q[7];
ry(2.716186540460395) q[5];
ry(1.96965845409785) q[6];
cx q[5],q[6];
ry(-2.917616694797463) q[5];
ry(-1.9694426652888408) q[6];
cx q[5],q[6];
ry(-0.1588197852919526) q[0];
ry(1.2787953284104334) q[1];
cx q[0],q[1];
ry(-1.235391344224205) q[0];
ry(-0.08635280185283868) q[1];
cx q[0],q[1];
ry(0.5222784840096626) q[2];
ry(-0.5094013043394224) q[3];
cx q[2],q[3];
ry(2.999966890077757) q[2];
ry(0.5900030653169904) q[3];
cx q[2],q[3];
ry(2.8472867494000944) q[4];
ry(2.823373406749642) q[5];
cx q[4],q[5];
ry(-1.0452583022290742) q[4];
ry(2.2547377994841717) q[5];
cx q[4],q[5];
ry(-0.2705707853131676) q[6];
ry(-0.7358434035560678) q[7];
cx q[6],q[7];
ry(-1.4768811428865163) q[6];
ry(-0.730056079386098) q[7];
cx q[6],q[7];
ry(2.599675826415335) q[0];
ry(1.2918758585720416) q[2];
cx q[0],q[2];
ry(-0.4431896469953438) q[0];
ry(2.6074822309919012) q[2];
cx q[0],q[2];
ry(0.925767582827454) q[2];
ry(-2.7351471031108554) q[4];
cx q[2],q[4];
ry(-2.595797965734992) q[2];
ry(-2.3503616357373724) q[4];
cx q[2],q[4];
ry(1.0258472915904784) q[4];
ry(-0.5638994387229213) q[6];
cx q[4],q[6];
ry(1.0142208285347456) q[4];
ry(-0.5672323297983716) q[6];
cx q[4],q[6];
ry(-1.569381445359527) q[1];
ry(0.6115762060658209) q[3];
cx q[1],q[3];
ry(-1.219907642981468) q[1];
ry(1.3996846900254798) q[3];
cx q[1],q[3];
ry(0.8788402460448459) q[3];
ry(2.4380891060917107) q[5];
cx q[3],q[5];
ry(-0.5625904931589412) q[3];
ry(0.06725274679265696) q[5];
cx q[3],q[5];
ry(1.2228766275204546) q[5];
ry(-0.8693339239669343) q[7];
cx q[5],q[7];
ry(-2.7335007271143974) q[5];
ry(-0.553852832993198) q[7];
cx q[5],q[7];
ry(-2.337025166196783) q[0];
ry(1.7753611180032571) q[3];
cx q[0],q[3];
ry(-0.1102203178293397) q[0];
ry(1.1270259888831466) q[3];
cx q[0],q[3];
ry(-2.9076412138311007) q[1];
ry(0.4479367257286618) q[2];
cx q[1],q[2];
ry(0.6570544002250402) q[1];
ry(-0.6309107308619042) q[2];
cx q[1],q[2];
ry(-3.0359016679883126) q[2];
ry(-0.0028180195624900256) q[5];
cx q[2],q[5];
ry(-2.176388693229284) q[2];
ry(-2.6684426534884165) q[5];
cx q[2],q[5];
ry(2.340205403418372) q[3];
ry(1.4688413822878292) q[4];
cx q[3],q[4];
ry(-2.87352315618079) q[3];
ry(-2.9780231287604115) q[4];
cx q[3],q[4];
ry(0.7726231831153032) q[4];
ry(3.0984959996723225) q[7];
cx q[4],q[7];
ry(0.1376486008216009) q[4];
ry(-2.8570172518547463) q[7];
cx q[4],q[7];
ry(-0.3476520120156499) q[5];
ry(-2.7546455776486343) q[6];
cx q[5],q[6];
ry(-1.8268890654562604) q[5];
ry(2.057624086166519) q[6];
cx q[5],q[6];
ry(-0.3550062751562813) q[0];
ry(-2.143036648987735) q[1];
cx q[0],q[1];
ry(3.1319643843095957) q[0];
ry(-1.572501207746476) q[1];
cx q[0],q[1];
ry(1.5567322474515164) q[2];
ry(2.0419389792000593) q[3];
cx q[2],q[3];
ry(-1.565518327717566) q[2];
ry(-2.605512119628655) q[3];
cx q[2],q[3];
ry(-0.15276114298239385) q[4];
ry(-2.6673902480247924) q[5];
cx q[4],q[5];
ry(-0.28364269656772034) q[4];
ry(-2.5367532119372034) q[5];
cx q[4],q[5];
ry(-1.393649150479811) q[6];
ry(-1.2177510686187392) q[7];
cx q[6],q[7];
ry(0.24094171714957874) q[6];
ry(-1.3632730297572797) q[7];
cx q[6],q[7];
ry(-0.06806169501244308) q[0];
ry(1.8757290779662055) q[2];
cx q[0],q[2];
ry(1.2436731999920445) q[0];
ry(-1.2300027068217263) q[2];
cx q[0],q[2];
ry(2.5642722613653546) q[2];
ry(0.9078398171456942) q[4];
cx q[2],q[4];
ry(-2.0173326907969926) q[2];
ry(3.046338928485465) q[4];
cx q[2],q[4];
ry(-2.9305893443123576) q[4];
ry(-1.2465848653712683) q[6];
cx q[4],q[6];
ry(2.609719689159053) q[4];
ry(-0.5651297807041341) q[6];
cx q[4],q[6];
ry(2.9751823480476878) q[1];
ry(-0.26931477061735515) q[3];
cx q[1],q[3];
ry(1.64190179606139) q[1];
ry(-1.336289725941094) q[3];
cx q[1],q[3];
ry(0.12162919885726886) q[3];
ry(-2.7234703562120655) q[5];
cx q[3],q[5];
ry(1.470389234937169) q[3];
ry(-0.7228760154875165) q[5];
cx q[3],q[5];
ry(-0.4558275161321204) q[5];
ry(1.4486980010556585) q[7];
cx q[5],q[7];
ry(3.0219032608491543) q[5];
ry(2.3161627396576607) q[7];
cx q[5],q[7];
ry(-0.4512905814067425) q[0];
ry(2.9739652246622104) q[3];
cx q[0],q[3];
ry(-1.2284891641727285) q[0];
ry(1.8562138949070925) q[3];
cx q[0],q[3];
ry(1.4832627275459187) q[1];
ry(-2.2047088150179874) q[2];
cx q[1],q[2];
ry(-0.6624787890547497) q[1];
ry(-0.241868775053252) q[2];
cx q[1],q[2];
ry(-1.2942192364552527) q[2];
ry(-2.085658099507466) q[5];
cx q[2],q[5];
ry(-2.126865155559807) q[2];
ry(1.339501574564286) q[5];
cx q[2],q[5];
ry(-1.1956369131262132) q[3];
ry(-0.893610641076243) q[4];
cx q[3],q[4];
ry(0.0686886489729277) q[3];
ry(-0.33116454578267845) q[4];
cx q[3],q[4];
ry(-1.0585028226829956) q[4];
ry(0.20594096923566071) q[7];
cx q[4],q[7];
ry(-0.5567936596784548) q[4];
ry(0.9669073417440731) q[7];
cx q[4],q[7];
ry(1.5669316160980087) q[5];
ry(-3.097275139317417) q[6];
cx q[5],q[6];
ry(-2.432766144400378) q[5];
ry(-1.974314038395047) q[6];
cx q[5],q[6];
ry(2.7602743154538167) q[0];
ry(1.6451528955013055) q[1];
ry(-1.4080007957702874) q[2];
ry(3.078147643417778) q[3];
ry(-0.016879450451865452) q[4];
ry(-0.4653375579381827) q[5];
ry(3.0431675138329997) q[6];
ry(0.3650746497787953) q[7];