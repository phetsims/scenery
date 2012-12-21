// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
	phet.scene.Node = function( name ) {
		this.children = [];
		this.transform = new phet.math.Transform3();
	}

	var Node = phet.scene.Node;

	Node.prototype = {
		constructor: Node
	};
})();