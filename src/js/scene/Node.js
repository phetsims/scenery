// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
	phet.scene.Node = function( name ) {
		this.children = [];
		this.transform = new phet.math.Transform3();
        this.parent = null;
        this.visible = true;
	}

	var Node = phet.scene.Node;
    var Matrix3 = phet.math.Matrix3;
    
    Node.RenderState = function () {
        this.transform = new phet.math.Transform3();
    };

	Node.prototype = {
		constructor: Node,
        
        render: function( args ) {
            if ( !this.transform.isIdentity() ) {
                args.transform.append( this.transform.matrix );
            }

            this.preRender( args );

            // TODO: consider allowing render passes here?
            this.renderSelf( args );
            this.renderChildren( args );

            this.postRender( args );

            if ( !this.transform.isIdentity() ) {
                args.transform.append( this.transform.inverse );
            }
        },
        
        renderSelf: function ( args ) {

        },

        renderChildren: function ( args ) {
            for ( var i = 0; i < this.children.length; i++ ) {
                this.children[i].render( args );
            }
        },

        preRender: function ( args ) {
            
        },

        postRender: function ( args ) {
            
        },

        addChild: function ( node ) {
            phet.assert( node !== null && node !== undefined );
            if ( this.isChild( node ) ) {
                return;
            }
            if ( node.parent !== null ) {
                node.parent.removeChild( node );
            }
            node.parent = this;
            this.children.push( node );
        },

        removeChild: function ( node ) {
            phet.assert( this.isChild( node ) );

            node.parent = null;
            this.children.splice( this.children.indexOf( node ), 1 );
        },

        hasParent: function () {
            return this.parent !== null && this.parent !== undefined;
        },

        detach: function () {
            if ( this.hasParent() ) {
                this.parent.removeChild( this );
            }
        },

        isChild: function ( potentialChild ) {
            phet.assert( (potentialChild.parent === this ) === (this.children.indexOf( potentialChild ) != -1) );
            return potentialChild.parent === this;
        },

        translate: function ( x, y ) {
            this.transform.append( Matrix3.translation( x, y ) );
        },

        // scale( s ) is also supported
        scale: function ( x, y ) {
            this.transform.append( Matrix3.scaling( x, y ) );
        },

        rotate: function ( angle ) {
            this.transform.append( Matrix3.rotation2( angle ) );
        }
	};
})();