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
    
	Node.prototype = {
		constructor: Node,
        
        render: function( state ) {
            if ( !this.transform.isIdentity() ) {
                //state.transform.append( this.transform.matrix );
                state.context.transform( 
                    // inlined array entries
                    this.transform.matrix.entries[0],
                    this.transform.matrix.entries[1],
                    this.transform.matrix.entries[3],
                    this.transform.matrix.entries[4],
                    this.transform.matrix.entries[6],
                    this.transform.matrix.entries[7]
                );
            }

            this.preRender( state );

            // TODO: consider allowing render passes here?
            this.renderSelf( state );
            this.renderChildren( state );

            this.postRender( state );

            if ( !this.transform.isIdentity() ) {
                //state.transform.append( this.transform.inverse );
                state.context.transform( 
                    // inlined array entries
                    this.transform.inverse.entries[0],
                    this.transform.inverse.entries[1],
                    this.transform.inverse.entries[3],
                    this.transform.inverse.entries[4],
                    this.transform.inverse.entries[6],
                    this.transform.inverse.entries[7]
                );
            }
        },
        
        renderSelf: function ( state ) {

        },

        renderChildren: function ( state ) {
            for ( var i = 0; i < this.children.length; i++ ) {
                this.children[i].render( state );
            }
        },

        preRender: function ( state ) {
            
        },

        postRender: function ( state ) {
            
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