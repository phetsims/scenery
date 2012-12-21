// Copyright 2002-2012, University of Colorado

/**
 * WebGL scenegraph node
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.webgl = phet.webgl || {};

// create a new scope
(function () {
    phet.webgl.GLNode = function () {
        this.parent = null;
        this.children = [];
        this.visible = true;
        this.transform = new phet.math.Transform4();

        this.wrappers = []; // rendering functions executed before and after the main's execution

        // TODO: consider adding in enable/disable flags! Does WebGL have that attribute stack?
    };

    var GLNode = phet.webgl.GLNode;
    var Matrix4 = phet.math.Matrix4;

    // TODO: rename args to renderState throughout?
    GLNode.RenderState = function () {
        this.transform = new phet.math.Transform4();

        this.transformAttribute = null;

        this.inverseTransposeAttribute = null; // used for mapping normals

        this.positionAttribute = null;

        this.normalAttribute = null;

        this.textureCoordinateAttribute = null;
    };

    GLNode.prototype = {
        constructor: GLNode,

        // required args to contain:
        //    transform: Transform4 indicating current transform
        render: function ( args ) {
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
            // TODO: state control here!
            for ( var i = 0; i < this.wrappers.length; i++ ) {
                this.wrappers[i].preRender( args );
            }
        },

        postRender: function ( args ) {
            for ( var i = 0; i < this.wrappers.length; i++ ) {
                this.wrappers[i].postRender( args );
            }
            // TODO: state control here!
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

        translate: function ( x, y, z ) {
            this.transform.append( Matrix4.translation( x, y, z ) );
        },

        // scale( s ) is also supported
        scale: function ( x, y, z ) {
            this.transform.append( Matrix4.scaling( x, y, z ) );
        },

        rotate: function ( axis, angle ) {
            this.transform.append( Matrix4.rotationAxisAngle( axis, angle ) );
        }
    };
})();
