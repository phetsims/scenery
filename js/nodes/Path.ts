// Copyright 2013-2023, University of Colorado Boulder

/**
 * A Path draws a Shape with a specific type of fill and stroke. Mixes in Paintable.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import { Shape } from '../../../kite/js/imports.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { CanvasContextWrapper, CanvasSelfDrawable, Instance, TPathDrawable, Node, NodeOptions, Paint, Paintable, PAINTABLE_DRAWABLE_MARK_FLAGS, PAINTABLE_OPTION_KEYS, PaintableOptions, PathCanvasDrawable, PathSVGDrawable, Renderer, scenery, SVGSelfDrawable } from '../imports.js';
import optionize, { combineOptions } from '../../../phet-core/js/optionize.js';

const PATH_OPTION_KEYS = [
  'boundsMethod',
  'shape'
];

const DEFAULT_OPTIONS = {
  shape: null,
  boundsMethod: 'accurate' as const
};

export type PathBoundsMethod = 'accurate' | 'unstroked' | 'tightPadding' | 'safePadding' | 'none';

type SelfOptions = {
  /**
   * This sets the shape of the Path, which determines the shape of its appearance. It should generally not be called
   * on Path subtypes like Line, Rectangle, etc.
   *
   * NOTE: When you create a Path with a shape in the constructor, this function will be called.
   *
   * The valid parameter types are:
   * - Shape: (from Kite), normally used.
   * - string: Uses the SVG Path format, see https://www.w3.org/TR/SVG/paths.html (the PATH part of <path d="PATH"/>).
   *           This will immediately be converted to a Shape object, and getShape() or equivalents will return the new
   *           Shape object instead of the original string.
   * - null: Indicates that there is no Shape, and nothing is drawn. Usually used as a placeholder.
   *
   * NOTE: Be aware of the potential for memory leaks. If a Shape is not marked as immutable (with makeImmutable()),
   *       Path will add a listener so that it is updated when the Shape itself changes. If there is a listener
   *       added, keeping a reference to the Shape will also keep a reference to the Path object (and thus whatever
   *       Nodes are connected to the Path). For now, set path.shape = null if you need to release the reference
   *       that the Shape would have, or call dispose() on the Path if it is not needed anymore.
   */
  shape?: Shape | string | null;

  /**
   * Sets the bounds method for the Path. This determines how our (self) bounds are computed, and can particularly
   * determine how expensive to compute our bounds are if we have a stroke.
   *
   * There are the following options:
   * - 'accurate' - Always uses the most accurate way of getting bounds. Computes the exact stroked bounds.
   * - 'unstroked' - Ignores any stroke, just gives the filled bounds.
   *                 If there is a stroke, the bounds will be marked as inaccurate
   * - 'tightPadding' - Pads the filled bounds by enough to cover everything except mitered joints.
   *                     If there is a stroke, the bounds wil be marked as inaccurate.
   * - 'safePadding' - Pads the filled bounds by enough to cover all line joins/caps.
   * - 'none' - Returns Bounds2.NOTHING. The bounds will be marked as inaccurate.
   *            NOTE: It's important to provide a localBounds override if you use this option, so its bounds cover the
   *            Path's shape. (path.localBounds = ...)
   */
  boundsMethod?: PathBoundsMethod;
};
type ParentOptions = PaintableOptions & NodeOptions;
export type PathOptions = SelfOptions & ParentOptions;

export default class Path extends Paintable( Node ) {

  // The Shape used for displaying this Path.
  // NOTE: _shape can be lazily constructed in subtypes (may be null) if hasShape() is overridden to return true,
  //       like in Rectangle. This is because usually the actual Shape is already implied by other parameters,
  //       so it is best to not have to compute it on changes.
  // NOTE: Please use hasShape() to determine if we are actually drawing things, as it is subtype-safe.
  // (scenery-internal)
  public _shape: Shape | null;

  // This stores a stroked copy of the Shape which is lazily computed. This can be required for computing bounds
  // of a Shape with a stroke.
  private _strokedShape: Shape | null;

  // (scenery-internal)
  public _boundsMethod: PathBoundsMethod;

  // Used as a listener to Shapes for when they are invalidated. The listeners are not added if the Shape is
  // immutable, and if the Shape becomes immutable, then the listeners are removed.
  private readonly _invalidShapeListener: () => void;

  // Whether our shape listener is attached to a shape.
  private _invalidShapeListenerAttached: boolean;

  /**
   * Creates a Path with a given shape specifier (a Shape, a string in the SVG path format, or null to indicate no
   * shape).
   *
   * Path has two additional options (above what Node provides):
   * - shape: The actual Shape (or a string representing an SVG path, or null).
   * - boundsMethod: Determines how the bounds of a shape are determined.
   *
   * @param shape - The initial Shape to display. See setShape() for more details and documentation.
   * @param [providedOptions] - Path-specific options are documented in PATH_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  public constructor( shape: Shape | string | null, providedOptions?: PathOptions ) {
    assert && assert( providedOptions === undefined || Object.getPrototypeOf( providedOptions ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    if ( shape || providedOptions?.shape ) {
      assert && assert( !shape || !providedOptions?.shape, 'Do not define shape twice. Check constructor and providedOptions.' );
    }

    const options = optionize<PathOptions, SelfOptions, ParentOptions>()( {
      shape: shape,
      boundsMethod: DEFAULT_OPTIONS.boundsMethod
    }, providedOptions );

    super();

    this._shape = DEFAULT_OPTIONS.shape;
    this._strokedShape = null;
    this._boundsMethod = DEFAULT_OPTIONS.boundsMethod;
    this._invalidShapeListener = this.invalidateShape.bind( this );
    this._invalidShapeListenerAttached = false;

    this.invalidateSupportedRenderers();

    this.mutate( options );
  }

  public setShape( shape: Shape | string | null ): this {
    assert && assert( shape === null || typeof shape === 'string' || shape instanceof Shape,
      'A path\'s shape should either be null, a string, or a Shape' );

    if ( this._shape !== shape ) {
      // Remove Shape invalidation listener if applicable
      if ( this._invalidShapeListenerAttached ) {
        this.detachShapeListener();
      }

      if ( typeof shape === 'string' ) {
        // be content with setShape always invalidating the shape?
        shape = new Shape( shape );
      }
      this._shape = shape;
      this.invalidateShape();

      // Add Shape invalidation listener if applicable
      if ( this._shape && !this._shape.isImmutable() ) {
        this.attachShapeListener();
      }
    }
    return this;
  }

  public set shape( value: Shape | string | null ) { this.setShape( value ); }

  public get shape(): Shape | null { return this.getShape(); }

  /**
   * Returns the shape that was set for this Path (or for subtypes like Line and Rectangle, will return an immutable
   * Shape that is equivalent in appearance).
   *
   * It is best to generally assume modifications to the Shape returned is not supported. If there is no shape
   * currently, null will be returned.
   */
  public getShape(): Shape | null {
    return this._shape;
  }

  /**
   * Returns a lazily-created Shape that has the appearance of the Path's shape but stroked using the current
   * stroke style of the Path.
   *
   * NOTE: It is invalid to call this on a Path that does not currently have a Shape (usually a Path where
   *       the shape is set to null).
   */
  public getStrokedShape(): Shape {
    assert && assert( this.hasShape(), 'We cannot stroke a non-existing shape' );

    // Lazily compute the stroked shape. It should be set to null when we need to recompute it
    if ( !this._strokedShape ) {
      this._strokedShape = this.getShape()!.getStrokedShape( this._lineDrawingStyles );
    }

    return this._strokedShape;
  }

  /**
   * Returns a bitmask representing the supported renderers for the current configuration of the Path or subtype.
   *
   * Should be overridden by subtypes to either extend or restrict renderers, depending on what renderers are
   * supported.
   *
   * @returns - A bitmask that includes supported renderers, see Renderer for details.
   */
  protected getPathRendererBitmask(): number {
    // By default, Canvas and SVG are accepted.
    return Renderer.bitmaskCanvas | Renderer.bitmaskSVG;
  }

  /**
   * Triggers a check and update for what renderers the current configuration of this Path or subtype supports.
   * This should be called whenever something that could potentially change supported renderers happen (which can
   * be the shape, properties of the strokes or fills, etc.)
   */
  public override invalidateSupportedRenderers(): void {
    this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getPathRendererBitmask() );
  }

  /**
   * Notifies the Path that the Shape has changed (either the Shape itself has be mutated, a new Shape has been
   * provided).
   *
   * NOTE: This should not be called on subtypes of Path after they have been constructed, like Line, Rectangle, etc.
   */
  private invalidateShape(): void {
    this.invalidatePath();

    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      ( this._drawables[ i ] as unknown as TPathDrawable ).markDirtyShape(); // subtypes of Path may not have this, but it's called during construction
    }

    // Disconnect our Shape listener if our Shape has become immutable.
    // see https://github.com/phetsims/sun/issues/270#issuecomment-250266174
    if ( this._invalidShapeListenerAttached && this._shape && this._shape.isImmutable() ) {
      this.detachShapeListener();
    }
  }

  /**
   * Invalidates the node's self-bounds and any other recorded metadata about the outline or bounds of the Shape.
   *
   * This is meant to be used for all Path subtypes (unlike invalidateShape).
   */
  protected invalidatePath(): void {
    this._strokedShape = null;

    this.invalidateSelf(); // We don't immediately compute the bounds
  }

  /**
   * Attaches a listener to our Shape that will be called whenever the Shape changes.
   */
  private attachShapeListener(): void {
    assert && assert( !this._invalidShapeListenerAttached, 'We do not want to have two listeners attached!' );

    // Do not attach shape listeners if we are disposed
    if ( !this.isDisposed ) {
      this._shape!.invalidatedEmitter.addListener( this._invalidShapeListener );
      this._invalidShapeListenerAttached = true;
    }
  }

  /**
   * Detaches a previously-attached listener added to our Shape (see attachShapeListener).
   */
  private detachShapeListener(): void {
    assert && assert( this._invalidShapeListenerAttached, 'We cannot detach an unattached listener' );

    this._shape!.invalidatedEmitter.removeListener( this._invalidShapeListener );
    this._invalidShapeListenerAttached = false;
  }

  /**
   * Computes a more efficient selfBounds for our Path.
   *
   * @returns - Whether the self bounds changed.
   */
  protected override updateSelfBounds(): boolean {
    const selfBounds = this.hasShape() ? this.computeShapeBounds() : Bounds2.NOTHING;
    const changed = !selfBounds.equals( this.selfBoundsProperty._value );
    if ( changed ) {
      this.selfBoundsProperty._value.set( selfBounds );
    }
    return changed;
  }

  public setBoundsMethod( boundsMethod: PathBoundsMethod ): this {
    assert && assert( boundsMethod === 'accurate' ||
                      boundsMethod === 'unstroked' ||
                      boundsMethod === 'tightPadding' ||
                      boundsMethod === 'safePadding' ||
                      boundsMethod === 'none' );
    if ( this._boundsMethod !== boundsMethod ) {
      this._boundsMethod = boundsMethod;
      this.invalidatePath();

      this.rendererSummaryRefreshEmitter.emit(); // whether our self bounds are valid may have changed
    }
    return this;
  }

  public set boundsMethod( value: PathBoundsMethod ) { this.setBoundsMethod( value ); }

  public get boundsMethod(): PathBoundsMethod { return this.getBoundsMethod(); }

  /**
   * Returns the current bounds method. See setBoundsMethod for details.
   */
  public getBoundsMethod(): PathBoundsMethod {
    return this._boundsMethod;
  }

  /**
   * Computes the bounds of the Path (or subtype when overridden). Meant to be overridden in subtypes for more
   * efficient bounds computations (but this will work as a fallback). Includes the stroked region if there is a
   * stroke applied to the Path.
   */
  public computeShapeBounds(): Bounds2 {
    const shape = this.getShape();
    // boundsMethod: 'none' will return no bounds
    if ( this._boundsMethod === 'none' || !shape ) {
      return Bounds2.NOTHING;
    }
    else {
      // boundsMethod: 'unstroked', or anything without a stroke will then just use the normal shape bounds
      if ( !this.hasPaintableStroke() || this._boundsMethod === 'unstroked' ) {
        return shape.bounds;
      }
      else {
        // 'accurate' will always require computing the full stroked shape, and taking its bounds
        if ( this._boundsMethod === 'accurate' ) {
          return shape.getStrokedBounds( this.getLineStyles() );
        }
          // Otherwise we compute bounds based on 'tightPadding' and 'safePadding', the one difference being that
          // 'safePadding' will include whatever bounds necessary to include miters. Square line-cap requires a
        // slightly extended bounds in either case.
        else {
          let factor;
          // If miterLength (inside corner to outside corner) exceeds miterLimit * strokeWidth, it will get turned to
          // a bevel, so our factor will be based just on the miterLimit.
          if ( this._boundsMethod === 'safePadding' && this.getLineJoin() === 'miter' ) {
            factor = this.getMiterLimit();
          }
          else if ( this.getLineCap() === 'square' ) {
            factor = Math.SQRT2;
          }
          else {
            factor = 1;
          }
          return shape.bounds.dilated( factor * this.getLineWidth() / 2 );
        }
      }
    }
  }

  /**
   * Whether this Node's selfBounds are considered to be valid (always containing the displayed self content
   * of this node). Meant to be overridden in subtypes when this can change (e.g. Text).
   *
   * If this value would potentially change, please trigger the event 'selfBoundsValid'.
   */
  public override areSelfBoundsValid(): boolean {
    if ( this._boundsMethod === 'accurate' || this._boundsMethod === 'safePadding' ) {
      return true;
    }
    else if ( this._boundsMethod === 'none' ) {
      return false;
    }
    else {
      return !this.hasStroke(); // 'tightPadding' and 'unstroked' options
    }
  }

  /**
   * Returns our self bounds when our rendered self is transformed by the matrix.
   */
  public override getTransformedSelfBounds( matrix: Matrix3 ): Bounds2 {
    assert && assert( this.hasShape() );

    return ( this._stroke ? this.getStrokedShape() : this.getShape() )!.getBoundsWithTransform( matrix );
  }

  /**
   * Returns our safe self bounds when our rendered self is transformed by the matrix.
   */
  public override getTransformedSafeSelfBounds( matrix: Matrix3 ): Bounds2 {
    return this.getTransformedSelfBounds( matrix );
  }

  /**
   * Called from (and overridden in) the Paintable trait, invalidates our current stroke, triggering recomputation of
   * anything that depended on the old stroke's value. (scenery-internal)
   */
  public override invalidateStroke(): void {
    this.invalidatePath();

    this.rendererSummaryRefreshEmitter.emit(); // Stroke changing could have changed our self-bounds-validitity (unstroked/etc)

    super.invalidateStroke();
  }

  /**
   * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
   */
  public hasShape(): boolean {
    return !!this._shape;
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  protected override canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
    PathCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a SVG drawable for this Path. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createSVGDrawable( renderer: number, instance: Instance ): SVGSelfDrawable {
    // @ts-expect-error
    return PathSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Path. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-expect-error
    return PathCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Whether this Node itself is painted (displays something itself).
   */
  public override isPainted(): boolean {
    // Always true for Path nodes
    return true;
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Path's self content, or "outside".
   *
   * @param point - Considered to be in the local coordinate frame
   */
  public override containsPointSelf( point: Vector2 ): boolean {
    let result = false;
    if ( !this.hasShape() ) {
      return result;
    }

    // if this node is fillPickable, we will return true if the point is inside our fill area
    if ( this._fillPickable ) {
      result = this.getShape()!.containsPoint( point );
    }

    // also include the stroked region in the hit area if strokePickable
    if ( !result && this._strokePickable ) {
      result = this.getStrokedShape().containsPoint( point );
    }
    return result;
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   */
  public override getSelfShape(): Shape {
    return Shape.union( [
      ...( ( this.hasShape() && this._fillPickable ) ? [ this.getShape()! ] : [] ),
      ...( ( this.hasShape() && this._strokePickable ) ? [ this.getStrokedShape() ] : [] )
    ] );
  }

  /**
   * Returns whether this Path's selfBounds is intersected by the specified bounds.
   *
   * @param bounds - Bounds to test, assumed to be in the local coordinate frame.
   */
  public override intersectsBoundsSelf( bounds: Bounds2 ): boolean {
    // TODO: should a shape's stroke be included?
    return this._shape ? this._shape.intersectsBounds( bounds ) : false;
  }

  /**
   * Returns whether we need to apply a transform workaround for https://github.com/phetsims/scenery/issues/196, which
   * only applies when we have a pattern or gradient (e.g. subtypes of Paint).
   */
  private requiresSVGBoundsWorkaround(): boolean {
    if ( !this._stroke || !( this._stroke instanceof Paint ) || !this.hasShape() ) {
      return false;
    }

    const bounds = this.computeShapeBounds();
    return bounds.x * bounds.y === 0; // at least one of them was zero, so the bounding box has no area
  }

  /**
   * Override for extra information in the debugging output (from Display.getDebugHTML()). (scenery-internal)
   */
  public override getDebugHTMLExtras(): string {
    return this._shape ? ` (<span style="color: #88f" onclick="window.open( 'data:text/plain;charset=utf-8,' + encodeURIComponent( '${this._shape.getSVGPath()}' ) );">path</span>)` : '';
  }

  /**
   * Disposes the path, releasing shape listeners if needed (and preventing new listeners from being added).
   */
  public override dispose(): void {
    if ( this._invalidShapeListenerAttached ) {
      this.detachShapeListener();
    }

    super.dispose();
  }

  public override mutate( options?: PathOptions ): this {
    return super.mutate( options );
  }

  // Initial values for most Node mutator options
  public static readonly DEFAULT_PATH_OPTIONS = combineOptions<PathOptions>( {}, Node.DEFAULT_NODE_OPTIONS, DEFAULT_OPTIONS );
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Path.prototype._mutatorKeys = [ ...PAINTABLE_OPTION_KEYS, ...PATH_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * (scenery-internal)
 * @override
 */
Path.prototype.drawableMarkFlags = [ ...Node.prototype.drawableMarkFlags, ...PAINTABLE_DRAWABLE_MARK_FLAGS, 'shape' ];

scenery.register( 'Path', Path );
