// Copyright 2023, University of Colorado Boulder

/**
 * A Property that, if there is a unique path from one Node to another (A => root => B, or A => B, or B => A), will
 * contain the transformation matrix from A to B's coordinate frame (local coordinate frames by default).
 *
 * If there is no unique path, the value will be null.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import arrayDifference from '../../../phet-core/js/arrayDifference.js';
import optionize from '../../../phet-core/js/optionize.js';
import { AncestorNodesProperty, Node, scenery, Trail } from '../imports.js';

type CoordinateFrame = 'parent' | 'local';

export type MatrixBetweenPropertyOptions = {
  // Which coordinate frames we want to be converting from/to, for each node
  fromCoordinateFrame?: CoordinateFrame;
  toCoordinateFrame?: CoordinateFrame;
};

export default class MatrixBetweenProperty extends TinyProperty<Matrix3 | null> {

  private readonly fromAncestorsProperty: AncestorNodesProperty;
  private readonly toAncestorsProperty: AncestorNodesProperty;

  // When we have a unique connection with trails, this will contain the root node common to both.
  // NOTE: This might be one of the actual nodes itself.
  private rootNode: Node | null = null;

  // When we have a unique connection with trails, these will contain the trail to the root node
  private fromTrail: Trail | null = null;
  private toTrail: Trail | null = null;

  private readonly fromCoordinateFrame: CoordinateFrame;
  private readonly toCoordinateFrame: CoordinateFrame;

  // A set of nodes where we are listening to whether their transforms change
  private readonly listenedNodeSet: Set<Node> = new Set<Node>();
  private readonly _nodeTransformListener: () => void;

  public constructor( public readonly from: Node, public readonly to: Node, providedOptions?: MatrixBetweenPropertyOptions ) {

    const options = optionize<MatrixBetweenPropertyOptions>()( {
      fromCoordinateFrame: 'local',
      toCoordinateFrame: 'local'
    }, providedOptions );

    super( Matrix3.IDENTITY );

    this.fromCoordinateFrame = options.fromCoordinateFrame;
    this.toCoordinateFrame = options.toCoordinateFrame;

    // Identical matrices shouldn't trigger notifications
    this.useDeepEquality = true;

    this.fromAncestorsProperty = new AncestorNodesProperty( from );
    this.toAncestorsProperty = new AncestorNodesProperty( to );

    const updateListener = this.update.bind( this );
    this._nodeTransformListener = this.updateMatrix.bind( this );

    // We'll only trigger a full update when parents/ancestors change anywhere. Otherwise, we'll just do transform
    // changes with updateMatrix()
    this.fromAncestorsProperty.updateEmitter.addListener( updateListener );
    this.toAncestorsProperty.updateEmitter.addListener( updateListener );

    this.update();
  }

  private update(): void {
    // Track nodes (not just ancestors) here, in case one is an ancestor of the other
    // REVIEW: would it be more performant for below opperations if these were Sets?
    const fromNodes = [ ...this.fromAncestorsProperty.value, this.from ];
    const toNodes = [ ...this.toAncestorsProperty.value, this.to ];

    // Intersection (ancestors of from/to)
    const commonNodes = fromNodes.filter( a => toNodes.includes( a ) );

    let hasDAG = false;

    // We'll want to find all nodes that are common ancestors of both, BUT aren't superfluous (an ancestor of another
    // common ancestor, with no other paths).
    const rootNodes = commonNodes.filter( node => {
      const fromChildren = fromNodes.filter( aNode => node.hasChild( aNode ) );
      const toChildren = toNodes.filter( bNode => node.hasChild( bNode ) );

      const fromOnly: Node[] = [];
      const toOnly: Node[] = [];
      const both: Node[] = [];
      arrayDifference( fromChildren, toChildren, fromOnly, toOnly, both );

      const hasMultipleChildren = fromChildren.length > 1 || toChildren.length > 1;
      const hasUnsharedChild = fromOnly.length || toOnly.length;

      // If either has multiple children, AND we're not just a trivial ancestor of the root, we're in a DAG case
      if ( hasMultipleChildren && hasUnsharedChild ) {
        hasDAG = true;
      }

      const hasFromExclusive = fromOnly.length > 0 || this.from === node;
      const hasToExclusive = toOnly.length > 0 || this.to === node;

      return hasFromExclusive && hasToExclusive;
    } );

    if ( !hasDAG && rootNodes.length === 1 ) {
      // We have a root node, and should have unique trails!
      this.rootNode = rootNodes[ 0 ];

      // These should assert-error out if there is no unique trail for either
      this.fromTrail = this.from.getUniqueTrailTo( this.rootNode );
      this.toTrail = this.to.getUniqueTrailTo( this.rootNode );
    }
    else {
      this.rootNode = null;
      this.fromTrail = null;
      this.toTrail = null;
    }

    // Take note of the nodes we are listening to
    const nodeSet = new Set<Node>();
    this.fromTrail && this.fromTrail.nodes.forEach( node => nodeSet.add( node ) );
    this.toTrail && this.toTrail.nodes.forEach( node => nodeSet.add( node ) );

    // Add in new needed listeners
    nodeSet.forEach( node => {
      if ( !this.listenedNodeSet.has( node ) ) {
        this.addNodeListener( node );
      }
    } );

    // Remove listeners not needed anymore
    this.listenedNodeSet.forEach( node => {
      if ( !nodeSet.has( node ) && node !== this.from && node !== this.to ) {
        this.removeNodeListener( node );
      }
    } );

    this.updateMatrix();
  }

  private updateMatrix(): void {
    if ( this.rootNode && this.fromTrail && this.toTrail ) {

      // If one of these is an ancestor of the other AND the ancestor requests a "parent" coordinate frame, we'll need
      // to compute things to the next level up. Otherwise, we can ignore the root node's transform. This is NOT
      // just an optimization, since if we multiply in the root node's transform into both the fromMatrix and toMatrix,
      // we'll lead to numerical imprecision that could be avoided. With this, we can get precise/exact results, even
      // if there is a scale on the rootNode (imagine a ScreenView's transform).
      const fromSelf = this.fromTrail.nodes.length === 1;
      const toSelf = this.toTrail.nodes.length === 1;
      const useAncestorMatrix = ( fromSelf && this.fromCoordinateFrame === 'parent' ) || ( toSelf && this.toCoordinateFrame === 'parent' );

      // Instead of switching between 4 different matrix functions, we use the general form.
      const fromMatrix = this.fromTrail.getMatrixConcatenation(
        useAncestorMatrix ? 0 : 1,
        this.fromTrail.nodes.length - ( this.fromCoordinateFrame === 'parent' ? 1 : 0 )
      );
      const toMatrix = this.toTrail.getMatrixConcatenation(
        useAncestorMatrix ? 0 : 1,
        this.toTrail.nodes.length - ( this.toCoordinateFrame === 'parent' ? 1 : 0 )
      );

      // toPoint = toMatrix^-1 * fromMatrix * fromPoint
      this.value = toMatrix.inverted().timesMatrix( fromMatrix );
    }
    else {
      this.value = null;
    }
  }

  private addNodeListener( node: Node ): void {
    this.listenedNodeSet.add( node );
    node.transformEmitter.addListener( this._nodeTransformListener );
  }

  private removeNodeListener( node: Node ): void {
    this.listenedNodeSet.delete( node );
    node.transformEmitter.removeListener( this._nodeTransformListener );
  }

  public override dispose(): void {
    this.fromAncestorsProperty.dispose();
    this.toAncestorsProperty.dispose();

    super.dispose();
  }
}

scenery.register( 'MatrixBetweenProperty', MatrixBetweenProperty );
