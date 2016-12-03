// Copyright 2016-2016, University of Colorado Boulder

/**
 * A container Node that will align content within a specific bounding box.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  var ALIGNMENT_CONTAINER_OPTION_KEYS = [
    'alignBounds', // {Bounds2} - Available content will be aligned inside of these bounds. May be controlled by a group.
    'group', // {AlignmentGroup|null} - If provided, will keep the bounds region up-to-date with the group
    'xAlign', // {string} - 'left', 'center', or 'right', for horizontal positioning
    'yAlign', // {string} - 'top', 'center', or 'bottom', for vertical positioning
    'xMargin', // {number} - Non-negative margin that should exist on the left and right of the content
    'yMargin' // {number} - Non-negative margin that should exist on the top and bottom of the content.
  ];

  /**
   * An individual container for an alignment group. Will maintain its size to match that of the group by overriding
   * its localBounds, and will position its content inside its localBounds by respecting its alignment and margins.
   * @constructor
   * @public
   *
   * @param {Node} content - Node that was passed to createContainer(). We will change its position to keep it aligned.
   * @param {Object} [options] - See the _.extend below in AlignmentContainer for documentation. Also passed to Node.
   */
  function AlignmentContainer( content, options ) {
    // @private {Bounds2} - Instance kept and mutated.
    this._alignBounds = Bounds2.NOTHING.copy();

    // @private {string}
    this._xAlign = 'center';
    this._yAlign = 'center';

    // @private {number}
    this._xMargin = 0;
    this._yMargin = 0;

    // @private {Node}
    this._content = content;

    // @private {AlignmentGroup|null}
    this._group = null;

    // @private {function|null} - Callback for when bounds change
    this._contentBoundsListener = this.invalidateAlignment.bind( this );

    // Will be removed by dispose()
    this._content.on( 'bounds', this._contentBoundsListener );

    Node.call( this, _.extend( {}, options, {
      children: [ this._content ]
    } ) );
  }

  scenery.register( 'AlignmentContainer', AlignmentContainer );

  inherit( Node, AlignmentContainer, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: ALIGNMENT_CONTAINER_OPTION_KEYS.concat( Node.prototype._mutatorKeys ),

    /**
     * Triggers recomputation of the alignment
     * @public
     */
    invalidateAlignment: function() {
      // The group update will change our alignBounds if required.
      if ( this._group ) {
        this._group.onContainerContentResized( this );
      }

      // If the alignBounds didn't change, we'll still need to update our own layout
      this.updateLayout();
    },

    /**
     * Sets the alignment bounds (the bounds in which our content will be aligned).
     * @public
     *
     * @param {Bounds2} alignBounds
     * @returns {AlignmentContainer} - For chaining
     */
    setAlignBounds: function( alignBounds ) {
      assert && assert( alignBounds instanceof Bounds2 && !alignBounds.isEmpty() && alignBounds.isFinite(),
        'alignBounds should be a non-empty finite Bounds2' );

      if ( !this._alignBounds.equals( alignBounds ) ) {
        this._alignBounds.set( alignBounds );

        this.updateLayout();
      }
      return this;
    },
    set alignBounds( value ) { this.setAlignBounds( value ); },

    /**
     * Returns the current alignment bounds (see setAlignBounds for details).
     * @public
     *
     * @returns {Bounds2}
     */
    getAlignBounds: function() {
      return this._alignBounds;
    },
    get alignBounds() { return this.getAlignBounds(); },

    /**
     * Sets the attachment to an AlignmentGroup. When attached, the bounds of this container will be controlled by
     * the group.
     * @public
     *
     * @param {AlignmentGroup} group
     * @returns {AlignmentContainer} - For chaining
     */
    setGroup: function( group ) {
      assert && assert( group instanceof scenery.AlignmentGroup, 'group should be an AlignmentGroup' );

      if ( this._group !== group ) {
        this._group = group;

        this.invalidateAlignment();
      }

      return this;
    },
    set group( value ) { this.setGroup( value ); },

    /**
     * Returns the attached alignment group (if one exists), or null otherwise.
     * @public
     *
     * @returns {AlignmentGroup|null}
     */
    getGroup: function() {
      return this._group;
    },
    get group() { return this.getGroup(); },

    /**
     * Sets the horizontal alignment of this container.
     * @public
     *
     * Available values are 'left', 'center', or 'right'.
     *
     * @param {string} xAlign
     * @returns {AlignmentContainer} - For chaining
     */
    setXAlign: function( xAlign ) {
      assert && assert( xAlign === 'left' || xAlign === 'center' || xAlign === 'right',
        'xAlign should be one of: \'left\', \'center\', or \'right\'' );

      if ( this._xAlign !== xAlign ) {
        this._xAlign = xAlign;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set xAlign( value ) { this.setXAlign( value ); },

    /**
     * Returns the current horizontal alignment of this container.
     * @public
     *
     * @returns {string} - See setXAlign for values.
     */
    getXAlign: function() {
      return this._xAlign;
    },
    get xAlign() { return this.getXAlign(); },

    /**
     * Sets the vertical alignment of this container.
     * @public
     *
     * Available values are 'top', 'center', or 'bottom'.
     *
     * @param {string} yAlign
     * @returns {AlignmentContainer} - For chaining
     */
    setYAlign: function( yAlign ) {
      assert && assert( yAlign === 'top' || yAlign === 'center' || yAlign === 'bottom',
        'yAlign should be one of: \'top\', \'center\', or \'bottom\'' );

      if ( this._yAlign !== yAlign ) {
        this._yAlign = yAlign;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set yAlign( value ) { this.setYAlign( value ); },

    /**
     * Returns the current vertical alignment of this container.
     * @public
     *
     * @returns {string} - See setYAlign for values.
     */
    getYAlign: function() {
      return this._yAlign;
    },
    get yAlign() { return this.getYAlign(); },

    /**
     * Sets the horizontal margin of this container.
     * @public
     *
     * This margin is the minimum amount of horizontal space that will exist between the content and the left and
     * right sides of this container.
     *
     * @param {number} xMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setXMargin: function( xMargin ) {
      assert && assert( typeof xMargin === 'number' && isFinite( xMargin ) && xMargin >= 0,
        'xMargin should be a finite non-negative number' );

      if ( this._xMargin !== xMargin ) {
        this._xMargin = xMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set xMargin( value ) { this.setXMargin( value ); },

    /**
     * Returns the current horizontal margin of this container.
     * @public
     *
     * @returns {number} - See setXMargin for values.
     */
    getXMargin: function() {
      return this._xMargin;
    },
    get xMargin() { return this.getXMargin(); },

    /**
     * Sets the vertical margin of this container.
     * @public
     *
     * This margin is the minimum amount of vertical space that will exist between the content and the top and
     * bottom sides of this container.
     *
     * @param {number} yMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setYMargin: function( yMargin ) {
      assert && assert( typeof yMargin === 'number' && isFinite( yMargin ) && yMargin >= 0,
        'yMargin should be a finite non-negative number' );

      if ( this._yMargin !== yMargin ) {
        this._yMargin = yMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set yMargin( value ) { this.setYMargin( value ); },

    /**
     * Returns the current vertical margin of this container.
     * @public
     *
     * @returns {number} - See setYMargin for values.
     */
    getYMargin: function() {
      return this._yMargin;
    },
    get yMargin() { return this.getYMargin(); },

    /**
     * Returns the bounding box of this container's content. This will include any margins.
     * @private
     *
     * @returns {Bounds2}
     */
    getContentBounds: function() {
      return this._content.bounds.dilatedXY( this._xMargin, this._yMargin );
    },

    /**
     * Updates the layout of this alignment container.
     * @private
     */
    updateLayout: function() {
      // Ignore layout if we haven't been initialized properly yet, or if we get a bad result.
      if ( this._alignBounds.isEmpty() || !this._alignBounds.isFinite() ) {
        return;
      }

      this.localBounds = this._alignBounds;

      if ( this._xAlign === 'center' ) {
        this._content.centerX = this.localBounds.centerX;
      }
      else if ( this._xAlign === 'left' ) {
        this._content.left = this.localBounds.left + this._xMargin;
      }
      else if ( this._xAlign === 'right' ) {
        this._content.right = this.localBounds.right - this._xMargin;
      }
      else {
        assert && assert( 'Bad xAlign: ' + this._xAlign );
      }

      if ( this._yAlign === 'center' ) {
        this._content.centerY = this.localBounds.centerY;
      }
      else if ( this._yAlign === 'top' ) {
        this._content.top = this.localBounds.top + this._yMargin;
      }
      else if ( this._yAlign === 'bottom' ) {
        this._content.bottom = this.localBounds.bottom - this._yMargin;
      }
      else {
        assert && assert( 'Bad yAlign: ' + this._yAlign );
      }

      assert && assert( this.localBounds.containsBounds( this._content.bounds ),
        'All of our contents should be contained in our localBounds' );
    },

    /**
     * Disposes this container, so that the alignment group won't update this container, and won't use its bounds for
     * laying out the other containers.
     * @public
     */
    dispose: function() {
      this._content.off( 'bounds', this._contentBoundsListener );

      if ( this._group ) {
        this._group.disposeContainer( this );
      }
    }
  } );

  return AlignmentContainer;
} );
