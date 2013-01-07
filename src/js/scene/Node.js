// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
	phet.scene.Node = function( name ) {
        // user-editable properties
        this.visible = true;
        this.layerType = null; // null indicates there is no layer root here. otherwise should be a layer constructor function
        
        this.children = [];
        this.transform = new phet.math.Transform3();
        this.parent = null;
        
        // layer-specific data, currently updated in the rebuildLayers step
        this._layerBeforeRender = null; // for layer changes
        this._layerAfterRender = null;
	}

	var Node = phet.scene.Node;
    var Matrix3 = phet.math.Matrix3;
    
	Node.prototype = {
		constructor: Node,
        
        render: function( state ) {
            if( this._layerBeforeRender ) {
                state.switchToLayer( this._layerBeforeRender );
            }
            
            if ( !this.transform.isIdentity() ) {
                // TODO: consider a stack-based model for transforms?
                state.applyTransformationMatrix( this.transform.matrix );
            }

            this.preRender( state );

            // TODO: consider allowing render passes here?
            this.renderSelf( state );
            this.renderChildren( state );

            this.postRender( state );

            if ( !this.transform.isIdentity() ) {
                state.applyTransformationMatrix( this.transform.inverse );
            }
            
            if( this._layerAfterRender ) {
                state.switchToLayer( this._layerAfterRender );
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
        
        isLayerRoot: function() {
            return this.layerType != null;
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
        },
        
        rebuildLayers: function( main ) {
            // verify that this node is the effective root
            phet.assert( this.parent == null );
            
            // root needs to contain a layer type reference
            phet.assert( this.layerType != null );
            
            main.empty();
            
            function recursiveRebuild( node, baseLayerType ) {
                var hasLayer = node.layerType != null;
                if( !hasLayer ) {
                    // sanity checks, in case a layerType was removed
                    node._layerBeforeRender = null;
                    node._layerAfterRender = null;
                } else {
                    // create the layers for before/after
                    node._layerBeforeRender = new node.layerType( main );
                    node._layerAfterRender = new baseLayerType( main );
                    
                    // change the base layer type for the layer children
                    baseLayerType = node.layerType;
                }
                
                // for stacking, add the "before" layer before recursion
                if( hasLayer ) {
                    main.append( node._layerBeforeRender );
                }
                
                // handle layers for children
                _.each( node.children, function( child ) {
                    recursiveRebuild( child, baseLayerType );
                } );
                
                // and the "after" layer after recursion, on top of any child layers
                if( hasLayer ) {
                    main.append( node._layerAfterRender );
                }
            }
            
            // get the layer constructor
            var rootLayerType = this.layerType;
            
            // create the first layer (will be the only layer if no other nodes have a layerType)
            var startingLayer = new rootLayerType( main );
            main.append( startingLayer );
            this._layerBeforeRender = startingLayer;
            // no "after" layer needed for the root, since nothing is rendered after it
            this._layerAfterRender = null;
            
            _.each( this.children, function( child ) {
                recursiveRebuild( child, rootLayerType );
            } );
        }
	};
})();